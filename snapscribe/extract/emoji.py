"""Emoji placeholders for the classical OCR path.

PaddleOCR-family models are trained on text, not pictographs: an emoji is
either dropped or recognised as a nearby-looking character. Dropping it
silently loses meaning ("sure 👍" and "sure 😒" are not the same message), so
the colourful blobs the recogniser did *not* claim are found geometrically and
a placeholder is spliced into the text at the right place.

Policies (set per profile):

``placeholder``  insert ``[emoji]`` (the default)
``vision``       leave it to the vision model, which reads emoji directly
``drop``         ignore them
"""

from __future__ import annotations

import cv2
import numpy as np

from snapscribe.extract.base import ExtractResult, TextLine

MIN_SATURATION = 90
MIN_VALUE = 60


def emoji_boxes(
    crop: np.ndarray,
    text_boxes: list[tuple[int, int, int, int]],
    min_size: int = 18,
    max_size_fraction: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Colourful blobs in a crop that no recognised text line covers.

    Chat text is one flat colour; emoji are multi-hued. Saturation separates
    them cleanly in both light and dark themes.
    """
    if crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] >= MIN_SATURATION) & (hsv[:, :, 2] >= MIN_VALUE)).astype(np.uint8)
    if not mask.any():
        return []
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    max_size = int(min(crop.shape[:2]) * max_size_fraction) or min_size
    out: list[tuple[int, int, int, int]] = []
    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w < min_size or h < min_size:
            continue
        if w > max_size * 2 or h > max_size * 2:
            continue
        if not 0.4 <= w / h <= 2.5:
            continue
        box = (x, y, x + w, y + h)
        if any(_overlaps(box, t) > 0.35 for t in text_boxes):
            continue
        out.append(box)
    return out


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    area = max(1, (ax1 - ax0) * (ay1 - ay0))
    return inter / area


def insert_placeholders(
    result: ExtractResult,
    crop: np.ndarray,
    placeholder: str = "[emoji]",
    policy: str = "placeholder",
    scale: float = 1.0,
) -> ExtractResult:
    """Splice placeholders for unrecognised emoji into ``result.text``.

    Emoji and recognised text runs are merged into one stream of boxes, grouped
    into visual rows and read left to right — so an emoji at the start of a
    line comes out at the start of that line, not appended to the end of it.
    """
    if policy in ("drop", "vision"):
        return result
    text_boxes = [line.box for line in result.lines if line.text.strip()]
    boxes = emoji_boxes(crop, text_boxes)
    if not boxes:
        return result

    from snapscribe.extract.ocr import group_rows, rows_to_text

    segments = [line for line in result.lines if line.text.strip()]
    segments += [TextLine(text=placeholder, confidence=0.0, box=box) for box in boxes]
    rows = group_rows(segments)
    result.text = rows_to_text(rows)
    result.meta["emoji_boxes"] = len(boxes)
    return result
