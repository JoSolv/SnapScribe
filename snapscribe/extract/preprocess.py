"""Preparing a bubble crop for recognition.

Small, deliberate steps: give the text a margin, blank out any photo rows so
the recogniser does not hallucinate captions out of pixels, and upscale when
the glyphs are too small for the detector's receptive field.
"""

from __future__ import annotations

import cv2
import numpy as np

from snapscribe.core.models import Region
from snapscribe.imaging import dominant_color

MIN_TEXT_HEIGHT = 34
"""Below roughly this glyph height, OCR accuracy falls off a cliff."""
MAX_UPSCALE = 3.0
PAD = 12


def prepare_crop(
    image: np.ndarray,
    region: Region,
    blank_media: bool = True,
    pad: int = PAD,
    target_line_height: int = MIN_TEXT_HEIGHT,
) -> tuple[np.ndarray, float]:
    """Return the crop to recognise and the scale factor that was applied."""
    crop = region.crop(image).copy()
    if crop.size == 0:
        return crop, 1.0

    fill = region.fill_color or dominant_color(crop)
    fill_bgr = tuple(int(c) for c in fill)

    if blank_media:
        for span in region.meta.get("media_rows", []) or []:
            a, b = int(span[0]), int(span[1])
            a = max(0, min(a, crop.shape[0]))
            b = max(0, min(b, crop.shape[0]))
            if b > a:
                crop[a:b, :] = fill_bgr

    if pad:
        crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=fill_bgr)

    scale = 1.0
    line_h = estimate_line_height(crop, fill_bgr)
    if 0 < line_h < target_line_height:
        scale = min(MAX_UPSCALE, target_line_height / line_h)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop, scale


def estimate_line_height(crop: np.ndarray, fill: tuple[int, int, int]) -> int:
    """Median height of the ink rows — a proxy for glyph height."""
    diff = np.abs(crop.astype(np.int16) - np.asarray(fill, dtype=np.int16)).max(axis=2)
    ink = diff > 40
    rows = ink.sum(axis=1) > max(1, crop.shape[1] // 50)
    runs: list[int] = []
    count = 0
    for value in rows:
        if value:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    if not runs:
        return 0
    return int(np.median(runs))


def text_only_crop(image: np.ndarray, region: Region) -> np.ndarray:
    """Crop with media rows removed entirely (not blanked) — for the vision path,
    which reads a shorter image faster and is less tempted to describe photos."""
    crop = region.crop(image)
    spans = region.meta.get("media_rows") or []
    if not spans or crop.size == 0:
        return crop
    keep = np.ones(crop.shape[0], dtype=bool)
    for span in spans:
        a, b = int(span[0]), int(span[1])
        keep[max(0, a) : max(0, b)] = False
    if not keep.any():
        return crop[:1]
    return crop[keep]
