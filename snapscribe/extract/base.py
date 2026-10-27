"""The extractor interface.

Both paths — classical OCR and a vision model — take the *same* bubble crop and
return the *same* shape of result. That is what makes verification a simple
diff rather than a reconciliation problem: for every region there are two
opinions about the same pixels.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class TextLine:
    """One line of recognised text with its box in crop coordinates."""

    text: str
    confidence: float = 0.0
    box: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass
class ExtractResult:
    text: str = ""
    confidence: float = 0.0
    engine: str = ""
    lines: list[TextLine] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str = ""
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.error


@runtime_checkable
class Extractor(Protocol):
    """``extract(bubble_crop) -> (text, confidence)``, plus provenance."""

    name: str

    def available(self) -> bool:
        """False when the optional dependency or model is missing."""

    def extract(self, crop: np.ndarray) -> ExtractResult: ...


class BaseExtractor:
    """Shared plumbing: timing, error capture, batch with progress."""

    name = "base"

    def available(self) -> bool:  # pragma: no cover - overridden
        return True

    def describe(self) -> str:
        return self.name

    def _extract(self, crop: np.ndarray) -> ExtractResult:  # pragma: no cover
        raise NotImplementedError

    def extract(self, crop: np.ndarray) -> ExtractResult:
        started = time.perf_counter()
        try:
            result = self._extract(crop)
        except Exception as exc:
            result = ExtractResult(engine=self.name, error=f"{type(exc).__name__}: {exc}")
        result.engine = result.engine or self.name
        result.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    def extract_batch(
        self,
        crops: Sequence[np.ndarray],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[ExtractResult]:
        out: list[ExtractResult] = []
        total = len(crops)
        for i, crop in enumerate(crops):
            out.append(self.extract(crop))
            if progress is not None:
                progress(i + 1, total)
        return out


class NullExtractor(BaseExtractor):
    """Returns nothing. Keeps the pipeline runnable without the OCR extra."""

    name = "none"

    def available(self) -> bool:
        return True

    def _extract(self, crop: np.ndarray) -> ExtractResult:
        return ExtractResult(engine=self.name, text="", confidence=0.0)


_REGISTRY: dict[str, Callable[..., BaseExtractor]] = {}


def register(name: str, factory: Callable[..., BaseExtractor]) -> None:
    _REGISTRY[name] = factory


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


def get_extractor(name: str, **kwargs: object) -> BaseExtractor:
    """Build an extractor by name (``ocr``, ``vision``, ``none``)."""
    key = (name or "none").strip().lower()
    aliases = {"rapidocr": "ocr", "paddle": "ocr", "llm": "vision", "null": "none", "": "none"}
    key = aliases.get(key, key)
    if key not in _REGISTRY:
        raise KeyError(f"unknown extractor {name!r}; available: {', '.join(available_engines())}")
    return _REGISTRY[key](**kwargs)


def iter_available(names: Iterable[str], **kwargs: object) -> list[BaseExtractor]:
    out: list[BaseExtractor] = []
    for n in names:
        try:
            engine = get_extractor(n, **kwargs)
        except KeyError:
            continue
        if engine.available():
            out.append(engine)
    return out


register("none", NullExtractor)
