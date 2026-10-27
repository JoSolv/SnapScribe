"""Debug overlay: what the segmenter saw, drawn on the screenshot.

This is the single most useful artefact when tuning a profile, and the phase-7
GUI region view is built on the same region list.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from snapscribe.core.models import Region, RegionKind

#: BGR, because every overlay is drawn with OpenCV.
COLORS: dict[str, tuple[int, int, int]] = {
    "left": (80, 190, 60),  # green
    "right": (40, 140, 240),  # orange
    "center": (220, 120, 200),  # purple
    "unknown": (0, 200, 255),  # yellow
    "dropped": (60, 60, 230),  # red
    "header": (170, 170, 170),
    "footer": (170, 170, 170),
    "chat": (200, 200, 60),  # teal
    "attachment": (200, 80, 255),  # pink
    "avatar": (160, 160, 90),
}


def _color_for(region: Region) -> tuple[int, int, int]:
    if region.kind is RegionKind.DROPPED:
        return COLORS["dropped"]
    if region.kind is RegionKind.HEADER:
        return COLORS["header"]
    if region.kind is RegionKind.FOOTER:
        return COLORS["footer"]
    if region.kind is RegionKind.CHAT_AREA:
        return COLORS["chat"]
    if region.kind is RegionKind.AVATAR:
        return COLORS["avatar"]
    if region.kind in (RegionKind.SYSTEM, RegionKind.DATE_SEPARATOR):
        return COLORS["center"]
    return COLORS.get(region.side.value, COLORS["unknown"])


def _label(region: Region) -> str:
    if region.kind is RegionKind.DROPPED:
        return f"drop: {region.drop_reason}"
    if region.kind in (
        RegionKind.HEADER,
        RegionKind.FOOTER,
        RegionKind.CHAT_AREA,
        RegionKind.AVATAR,
    ):
        return region.kind.value
    bits = [f"#{region.order}", region.side.value]
    if region.kind is RegionKind.SYSTEM:
        bits.append("system")
    if region.meta.get("attachment"):
        bits.append("media")
    if region.confidence < 0.99:
        bits.append(f"{region.confidence:.2f}")
    return " ".join(bits)


def draw_overlay(
    image: np.ndarray,
    regions: list[Region],
    text: dict[str, str] | None = None,
    show_dropped: bool = True,
    scale: float = 1.0,
) -> np.ndarray:
    """Draw labelled boxes for every region on a copy of ``image``."""
    canvas = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(2, round(image.shape[1] / 540))
    font_scale = max(0.5, image.shape[1] / 1800)

    for region in regions:
        if region.kind is RegionKind.DROPPED and not show_dropped:
            continue
        color = _color_for(region)
        x0, y0, x1, y1 = region.box
        dashed = region.kind is RegionKind.DROPPED
        if dashed:
            _dashed_rect(canvas, (x0, y0), (x1 - 1, y1 - 1), color, thickness)
        else:
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, thickness)

        if region.kind in (RegionKind.BUBBLE, RegionKind.SYSTEM) and region.meta.get("media_rows"):
            for a, b in region.meta["media_rows"]:
                cv2.rectangle(
                    canvas,
                    (x0 + 2, y0 + int(a)),
                    (x1 - 3, y0 + int(b)),
                    COLORS["attachment"],
                    max(1, thickness - 1),
                )

        label = _label(region)
        if text and region.id in text:
            snippet = text[region.id].replace("\n", " ")[:48]
            label = f"{label}  {snippet}"
        _label_box(canvas, label, (x0, y0), color, font, font_scale, max(1, thickness - 1))

    if scale != 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return canvas


def _dashed_rect(img, pt1, pt2, color, thickness, dash=14):
    x0, y0 = pt1
    x1, y1 = pt2
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, thickness)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, thickness)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, thickness)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, thickness)


def _label_box(img, text, origin, color, font, font_scale, thickness):
    x, y = origin
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    y = max(y, th + base + 4)
    cv2.rectangle(img, (x, y - th - base - 4), (x + tw + 8, y), color, -1)
    cv2.putText(
        img, text, (x + 4, y - base - 2), font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA
    )


def save_overlay(
    path: str | Path,
    image: np.ndarray,
    regions: list[Region],
    text: dict[str, str] | None = None,
    show_dropped: bool = True,
    scale: float = 1.0,
) -> Path:
    from snapscribe.imaging import save_image

    return save_image(path, draw_overlay(image, regions, text, show_dropped, scale))


def legend_lines() -> list[str]:
    return [
        "green  = incoming (left)",
        "orange = outgoing (right)",
        "purple = system / date separator",
        "pink   = media rows inside a bubble",
        "red dashed = dropped (clipped, too small, too wide)",
        "grey   = header / footer chrome",
    ]


__all__ = ["COLORS", "draw_overlay", "legend_lines", "save_overlay"]
