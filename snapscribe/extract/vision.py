"""Vision-model extraction through a local Ollama server.

Design notes that matter for accuracy:

* The model is fed **one bubble crop at a time**, never a whole screenshot. It
  hallucinates far less on a small image, and the output lines up one-to-one
  with the OCR result so the two can be diffed per message.
* The prompt asks for the bubble *verbatim, including the timestamp*. Both
  engines must see the same thing; stripping the timestamp belongs downstream,
  where one rule applies to both.
* It is optional by construction. Without a server (or without a GPU / Apple
  Silicon to make it bearable) the pipeline runs OCR-only.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache

import cv2
import numpy as np

from snapscribe.extract.base import BaseExtractor, ExtractResult, TextLine, register

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("SNAPSCRIBE_VISION_MODEL", "qwen2.5vl:3b")

#: Candidates worth benchmarking on your own corpus before committing to one.
SUGGESTED_MODELS: tuple[str, ...] = (
    "qwen2.5vl:3b",
    "qwen2.5vl:7b",
    "moondream",
    "minicpm-v",
    "llama3.2-vision:11b",
    "granite3.2-vision",
)

PROMPT = (
    "This image is a single chat message bubble from a messaging app.\n"
    "Transcribe every character you can read, exactly as written, including the "
    "sender name if one is shown and the timestamp if one is shown.\n"
    "Keep the original line breaks. Keep emoji as emoji.\n"
    "If the bubble contains a photo or video, write [image] or [video] on its own "
    "line in place of it.\n"
    "Reply with the transcription only — no explanation, no quotes, no markdown."
)

_PREAMBLE_RE = re.compile(
    r"^\s*(?:(?:the\s+)?(?:image|bubble|message|text)\s+(?:shows|reads|says|contains)\s*:?\s*"
    r"|here(?:'s| is)\s+the\s+transcription\s*:?\s*"
    r"|transcription\s*:?\s*)",
    re.IGNORECASE,
)


def _encode_png(crop: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", crop)
    if not ok:  # pragma: no cover - cv2 encodes any uint8 BGR array
        raise ValueError("could not encode crop")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def clean_response(text: str) -> str:
    """Strip the chatter small models wrap around a transcription."""
    out = (text or "").strip()
    out = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", out).strip()
    previous = None
    while previous != out:
        previous = out
        out = _PREAMBLE_RE.sub("", out).strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'":
        inner = out[1:-1]
        if out[0] not in inner:
            out = inner
    return out.strip()


@lru_cache(maxsize=8)
def _server_models(host: str) -> tuple[str, ...]:
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=3) as fh:
            data = json.loads(fh.read())
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return ()
    return tuple(m.get("name", "") for m in data.get("models", []))


class VisionExtractor(BaseExtractor):
    name = "vision"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
        prompt: str = PROMPT,
        temperature: float = 0.0,
        **_: object,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.prompt = prompt
        self.temperature = temperature

    def available(self) -> bool:
        models = _server_models(self.host)
        if not models:
            return False
        base = self.model.split(":")[0]
        return any(m == self.model or m.split(":")[0] == base for m in models)

    def describe(self) -> str:
        state = "ready" if self.available() else "no server/model"
        return f"vision ({self.model} via {self.host}, {state})"

    def _extract(self, crop: np.ndarray) -> ExtractResult:
        if crop.size == 0:
            return ExtractResult(engine=self.name, text="", confidence=0.0)
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": self.prompt,
                "images": [_encode_png(crop)],
                "stream": False,
                "options": {"temperature": self.temperature},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as fh:
                data = json.loads(fh.read())
        except urllib.error.HTTPError as exc:
            return ExtractResult(
                engine=self.name, error=f"ollama HTTP {exc.code}: {exc.read()[:200]!r}"
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return ExtractResult(
                engine=self.name,
                error=f"could not reach ollama at {self.host}: {exc}",
            )
        text = clean_response(data.get("response", ""))
        lines = [TextLine(text=line, confidence=0.0) for line in text.splitlines() if line.strip()]
        return ExtractResult(
            engine=self.name,
            text=text,
            # Generative models do not report calibrated confidence. Reporting a
            # constant is honest; verification derives real signal from
            # agreement with OCR instead.
            confidence=0.5 if text else 0.0,
            lines=lines,
            meta={
                "model": self.model,
                "eval_count": data.get("eval_count"),
                "total_duration_ns": data.get("total_duration"),
            },
        )


register("vision", VisionExtractor)
