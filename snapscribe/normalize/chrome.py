"""Header/footer detection by pixel stability.

The app's chrome — status bar, title bar, composer — is the part of the screen
that does *not* change as you scroll. So: stack the screenshots, find the rows
that are the same in all of them, and take the runs touching the top and bottom
edges. No profile needed, no OCR, works on any app.

Two details make it survive real screenshots:

* a row counts as stable when *most* of its pixels agree, not all — the status
  bar clock changes between screenshots, and so does the battery icon;
* the detected run is then trimmed back to the last row that actually differs
  from the chat background, so blank wallpaper above the first bubble is not
  counted as chrome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from snapscribe.imaging import color_distance, dominant_color

DEFAULT_PIXEL_TOLERANCE = 12
"""Max per-channel difference still counted as "the same pixel"."""
DEFAULT_ROW_AGREEMENT = 0.85
"""Fraction of a row's pixels that must agree for the row to be stable."""
MAX_CHROME_FRACTION = 0.35
"""A header or footer taller than this is not chrome; fall back to the profile."""


@dataclass
class ChromeDetection:
    header_height: int = 0
    footer_height: int = 0
    method: str = "none"
    confidence: float = 0.0
    stability: list[float] = field(default_factory=list)
    """Per-row agreement, kept for the debug overlay."""
    notes: list[str] = field(default_factory=list)


def row_stability(
    images: Sequence[np.ndarray], pixel_tolerance: int = DEFAULT_PIXEL_TOLERANCE
) -> np.ndarray:
    """Fraction of each row's pixels that agree across every image."""
    if len(images) < 2:
        raise ValueError("row stability needs at least two images")
    stack = np.stack([im.astype(np.int16) for im in images])  # (N, H, W, 3)
    reference = np.median(stack, axis=0)
    deviation = np.abs(stack - reference).max(axis=3)  # (N, H, W) worst channel
    agree = (deviation <= pixel_tolerance).all(axis=0)  # (H, W)
    return agree.mean(axis=1)


def _leading_run(mask: np.ndarray) -> int:
    idx = np.flatnonzero(~mask)
    return int(idx[0]) if idx.size else int(mask.size)


def _trailing_run(mask: np.ndarray) -> int:
    idx = np.flatnonzero(~mask)
    return int(mask.size - 1 - idx[-1]) if idx.size else int(mask.size)


def _close_gaps(mask: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Bridge runs of up to ``max_gap`` unstable rows between stable ones."""
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            out[i:j] = True
        i = max(j, i + 1)
    return out


def _content_rows(
    image: np.ndarray, background: tuple[int, int, int], tolerance: int
) -> np.ndarray:
    """Rows containing at least a few pixels that are not the background."""
    diff = np.abs(image.astype(np.int16) - np.asarray(background, dtype=np.int16)).max(axis=2)
    non_bg = diff > tolerance
    min_px = max(3, image.shape[1] // 200)
    return non_bg.sum(axis=1) >= min_px


def detect_chrome_group(
    images: Sequence[np.ndarray],
    pixel_tolerance: int = DEFAULT_PIXEL_TOLERANCE,
    row_agreement: float = DEFAULT_ROW_AGREEMENT,
    bg_tolerance: int = 18,
) -> ChromeDetection:
    """Detect chrome shared by a group of same-size screenshots."""
    if len(images) < 2:
        return ChromeDetection(method="none", notes=["need at least two screenshots"])

    height = images[0].shape[0]
    stability = row_stability(images, pixel_tolerance)
    stable = _close_gaps(stability >= row_agreement)

    if stable.mean() > 0.9:
        return ChromeDetection(
            method="none",
            stability=[round(float(v), 4) for v in stability],
            notes=["screenshots are nearly identical; cannot tell chrome from content"],
        )

    header = _leading_run(stable)
    footer = _trailing_run(stable)

    cap = int(height * MAX_CHROME_FRACTION)
    notes: list[str] = []
    if header > cap:
        notes.append(f"header run {header}px exceeded {cap}px cap, clamped")
        header = cap
    if footer > cap:
        notes.append(f"footer run {footer}px exceeded {cap}px cap, clamped")
        footer = cap

    # Trim back to real chrome: measure the chat background from the middle of
    # the first image, then drop the plain-background rows at the inner edges.
    # The trim threshold is deliberately *stricter* than the segmenter's, so a
    # band that the segmenter would see as content (an app bar whose colour is
    # close to the wallpaper) always stays inside the chrome instead of
    # becoming a full-width phantom bubble.
    trim_tolerance = max(4, bg_tolerance - 6)
    middle = images[0][height // 3 : 2 * height // 3]
    background = dominant_color(middle)
    content = _content_rows(images[0], background, trim_tolerance)
    for other in images[1:]:
        content |= _content_rows(other, background, trim_tolerance)

    if header > 0:
        band = content[:header]
        last = np.flatnonzero(band)
        header = int(last[-1]) + 1 if last.size else 0
    if footer > 0:
        start = height - footer
        band = content[start:]
        first = np.flatnonzero(band)
        footer = height - (start + int(first[0])) if first.size else 0

    stable_in_header = float(stability[:header].mean()) if header else 1.0
    stable_in_footer = float(stability[height - footer :].mean()) if footer else 1.0
    confidence = min(stable_in_header, stable_in_footer)
    if header == 0 and footer == 0:
        confidence = 0.0
        notes.append("no stable rows at the image edges")

    return ChromeDetection(
        header_height=int(header),
        footer_height=int(footer),
        method="stable_rows",
        confidence=round(confidence, 4),
        stability=[round(float(v), 4) for v in stability],
        notes=notes,
    )


def detect_chrome(
    image: np.ndarray,
    header_fraction: float,
    footer_fraction: float,
) -> ChromeDetection:
    """Single-image fallback: trust the profile's fractions.

    Used when only one screenshot was imported, or when stability detection
    could not separate chrome from content.
    """
    h = image.shape[0]
    return ChromeDetection(
        header_height=round(h * header_fraction),
        footer_height=round(h * footer_fraction),
        method="profile",
        confidence=0.35,
        notes=["estimated from profile fractions, not measured"],
    )


def refine_with_profile(
    detection: ChromeDetection,
    image: np.ndarray,
    header_fraction: float,
    footer_fraction: float,
    max_relative_error: float = 0.6,
) -> ChromeDetection:
    """Sanity-check a measurement against what the profile expects.

    A measured header wildly different from the profile's usually means the
    screenshots were taken at different scroll positions of a *static* screen,
    or that there is only one distinct image. Trust the profile then.
    """
    if detection.method != "stable_rows":
        return detection
    h = image.shape[0]
    expected_header = h * header_fraction
    expected_footer = h * footer_fraction
    notes = list(detection.notes)
    out = ChromeDetection(
        header_height=detection.header_height,
        footer_height=detection.footer_height,
        method=detection.method,
        confidence=detection.confidence,
        stability=detection.stability,
        notes=notes,
    )
    if expected_header > 0:
        err = abs(detection.header_height - expected_header) / expected_header
        if err > max_relative_error:
            notes.append(
                f"measured header {detection.header_height}px differs from the profile's "
                f"~{expected_header:.0f}px by {err:.0%}"
            )
            out.confidence = min(out.confidence, 0.5)
    if expected_footer > 0:
        err = abs(detection.footer_height - expected_footer) / expected_footer
        if err > max_relative_error:
            notes.append(
                f"measured footer {detection.footer_height}px differs from the profile's "
                f"~{expected_footer:.0f}px by {err:.0%}"
            )
            out.confidence = min(out.confidence, 0.5)
    return out


def background_color(image: np.ndarray, chrome: ChromeDetection) -> tuple[int, int, int]:
    """Chat-area background colour, measured away from the chrome."""
    h = image.shape[0]
    top = chrome.header_height
    bottom = h - chrome.footer_height
    band = image[top:bottom] if bottom > top else image
    return dominant_color(band)


def color_matches(a: tuple[int, int, int], b: tuple[int, int, int], tolerance: float) -> bool:
    return color_distance(a, b) <= tolerance
