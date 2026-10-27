"""Classical OCR via RapidOCR (PaddleOCR's models on ONNX Runtime).

RapidOCR was chosen over PaddleOCR itself for install weight: same detection
and recognition models, no paddlepaddle, ~80 MB instead of ~1 GB, and it runs
on CPU at a usable speed. The wrapper tolerates both the 1.x
(``rapidocr_onnxruntime``) and 3.x (``rapidocr``) APIs, whose return shapes
differ.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import numpy as np

from snapscribe.extract.base import BaseExtractor, ExtractResult, TextLine, register


@lru_cache(maxsize=2)
def _load_engine(lang: str = "") -> tuple[Any, str] | None:
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]

        return RapidOCR(), "rapidocr_onnxruntime"
    except ImportError:
        pass
    try:
        from rapidocr import RapidOCR  # type: ignore[import-not-found]

        return RapidOCR(), "rapidocr"
    except ImportError:
        return None


def _box_to_rect(box: Any) -> tuple[int, int, int, int]:
    pts = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return (int(x0), int(y0), int(x1), int(y1))


def _normalise_result(raw: Any) -> list[TextLine]:
    """Flatten whichever shape the installed RapidOCR returned."""
    lines: list[TextLine] = []
    if raw is None:
        return lines
    # 1.x: (list_of_[box, text, score], elapsed)
    if isinstance(raw, tuple) and len(raw) == 2 and isinstance(raw[0], list | type(None)):
        raw = raw[0] or []
    if isinstance(raw, list):
        for item in raw:
            if not item:
                continue
            box, text, score = item[0], item[1], item[2]
            lines.append(TextLine(text=str(text), confidence=float(score), box=_box_to_rect(box)))
        return lines
    # 3.x: an object with .txts / .scores / .boxes
    texts = getattr(raw, "txts", None)
    if texts:
        scores = getattr(raw, "scores", None) or [0.0] * len(texts)
        boxes = getattr(raw, "boxes", None)
        for i, text in enumerate(texts):
            box = _box_to_rect(boxes[i]) if boxes is not None else (0, 0, 0, 0)
            lines.append(TextLine(text=str(text), confidence=float(scores[i]), box=box))
    return lines


def group_rows(lines: list[TextLine], overlap: float = 0.4) -> list[list[TextLine]]:
    """Group recognised boxes into visual rows.

    A detector emits one box per *text run*, not per line: an emoji or a wide
    gap in the middle of a sentence splits it into two boxes at the same
    height. Grouping by vertical overlap and joining with a space — rather than
    treating every box as its own line — is what keeps "Sending that to my
    cardiologist" from coming out as two lines.
    """
    if not lines:
        return []
    ordered = sorted(lines, key=lambda line: (line.box[1], line.box[0]))
    rows: list[list[TextLine]] = [[ordered[0]]]
    for line in ordered[1:]:
        row = rows[-1]
        ry0 = min(item.box[1] for item in row)
        ry1 = max(item.box[3] for item in row)
        y0, y1 = line.box[1], line.box[3]
        inter = max(0, min(y1, ry1) - max(y0, ry0))
        height = max(1, min(y1 - y0, ry1 - ry0))
        if inter / height >= overlap:
            row.append(line)
        else:
            rows.append([line])
    return [_dedupe_row(sorted(row, key=lambda line: line.box[0])) for row in rows]


def _dedupe_row(row: list[TextLine], overlap: float = 0.6) -> list[TextLine]:
    """Drop boxes that mostly repeat a neighbour.

    Text detectors sometimes emit a short box *and* a longer box covering the
    same words ("Sending t" alongside "Sending that to my cardiologist"), which
    would otherwise be concatenated into a stutter.
    """
    kept: list[TextLine] = []
    for line in row:
        x0, x1 = line.box[0], line.box[2]
        width = max(1, x1 - x0)
        replaced = False
        drop = False
        for i, other in enumerate(kept):
            ox0, ox1 = other.box[0], other.box[2]
            inter = max(0, min(x1, ox1) - max(x0, ox0))
            if inter / min(width, max(1, ox1 - ox0)) < overlap:
                continue
            if len(line.text.strip()) > len(other.text.strip()):
                kept[i] = line
                replaced = True
            else:
                drop = True
            break
        if not replaced and not drop:
            kept.append(line)
    return kept


_LONE_PUNCT = re.compile(r"^[^\w\s]$")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > 0


def _repair_boundary(left: str, right: str) -> tuple[str, str]:
    """Undo the character a detector duplicates where it split a line.

    When two boxes on the same row overlap horizontally, the glyphs in the
    overlap are read twice: "Morning! Did y" + "you manage to…", or
    "Fixeda" + "and uploaded". Trimming the repeated characters off the shorter
    side reconstructs the original line.
    """
    a_words, b_words = left.split(), right.split()
    if not a_words or not b_words:
        return left, right
    last, first = a_words[-1], b_words[0]

    if _LONE_PUNCT.match(first) and len(b_words) > 1:
        return left, " ".join(b_words[1:])
    if _LONE_PUNCT.match(last) and len(a_words) > 1:
        return " ".join(a_words[:-1]), right

    if last.casefold() == first.casefold():
        return left, " ".join(b_words[1:])

    limit = min(4, len(last), len(first))
    for k in range(limit, 0, -1):
        if last[-k:].casefold() != first[:k].casefold():
            continue
        if len(last) <= k:
            return " ".join(a_words[:-1]), right
        if len(first) <= k:
            return left, " ".join(b_words[1:])
        # Both sides are real words: the repeat belongs to whichever is
        # left dangling by it.
        return " ".join([*a_words[:-1], last[:-k]]), right
    return left, right


def rows_to_text(rows: list[list[TextLine]]) -> str:
    out = []
    for row in rows:
        parts: list[str] = []
        boxes: list[tuple[int, int, int, int]] = []
        for item in row:
            text = item.text.strip()
            if not text:
                continue
            if parts and _boxes_overlap(boxes[-1], item.box):
                parts[-1], text = _repair_boundary(parts[-1], text)
                if not parts[-1]:
                    parts.pop()
                    boxes.pop()
            if text:
                parts.append(text)
                boxes.append(item.box)
        joined = " ".join(p for p in parts if p)
        if joined:
            out.append(joined)
    return "\n".join(out)


def sort_lines(lines: list[TextLine], line_tolerance: int = 12) -> list[TextLine]:
    """Reading order: top to bottom, left to right within a visual row."""
    return [line for row in group_rows(lines) for line in row]


def aggregate_confidence(lines: list[TextLine]) -> float:
    """Character-weighted mean, so a confident "ok" does not outvote a long line."""
    weight = sum(max(1, len(line.text)) for line in lines)
    if not weight:
        return 0.0
    return sum(line.confidence * max(1, len(line.text)) for line in lines) / weight


class OcrExtractor(BaseExtractor):
    name = "ocr"

    def __init__(self, lang: str = "", join: str = "\n", **_: object) -> None:
        self.lang = lang
        self.join = join

    def available(self) -> bool:
        return _load_engine(self.lang) is not None

    def describe(self) -> str:
        loaded = _load_engine(self.lang)
        return f"ocr ({loaded[1]})" if loaded else "ocr (not installed)"

    def _extract(self, crop: np.ndarray) -> ExtractResult:
        loaded = _load_engine(self.lang)
        if loaded is None:
            return ExtractResult(
                engine=self.name,
                error="RapidOCR is not installed; `uv pip install -e '.[ocr]'`",
            )
        engine, backend = loaded
        if crop.size == 0:
            return ExtractResult(engine=self.name, text="", confidence=0.0)
        raw = engine(crop)
        rows = group_rows(_normalise_result(raw))
        lines = [line for row in rows for line in row]
        text = rows_to_text(rows)
        return ExtractResult(
            engine=self.name,
            text=text,
            confidence=round(aggregate_confidence(lines), 4),
            lines=lines,
            meta={"backend": backend},
        )


register("ocr", OcrExtractor)
