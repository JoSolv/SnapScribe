"""Import-time image quality checks.

Cheap signals that tell the user *why* a transcript came out badly, and that
the photo path (phase 8) uses to pick a master image.
"""

from __future__ import annotations

import numpy as np

from snapscribe.core.models import Quality
from snapscribe.imaging import dominant_color, glare_ratio, laplacian_variance

BLUR_THRESHOLD = 90.0
"""Laplacian variance below this is blurry for UI text, which is high-contrast."""
GLARE_THRESHOLD = 0.035
MIN_SHORT_EDGE = 600


UI_FLATNESS = 0.2
"""Share of an image covered by one colour above which it is a rendered UI."""


def looks_rendered(image: np.ndarray) -> bool:
    """True for a native screenshot: one flat colour covers much of it.

    Blur and glare are properties of a photograph. Running those checks on a
    screenshot produces nonsense — a light-theme chat is 30% pure white, which
    the glare detector reads as a reflection.
    """
    if image.size == 0:
        return False
    colour = np.asarray(dominant_color(image), dtype=np.int16)
    diff = np.abs(image.astype(np.int16) - colour).max(axis=2)
    return float((diff <= 6).mean()) >= UI_FLATNESS


def assess_quality(
    image: np.ndarray,
    blur_threshold: float = BLUR_THRESHOLD,
    glare_threshold: float = GLARE_THRESHOLD,
    force: bool = False,
) -> Quality:
    """Blur/glare signals. Skipped for images that are plainly rendered UI."""
    if not force and looks_rendered(image):
        q = Quality()
        short_edge = min(image.shape[:2])
        if short_edge < MIN_SHORT_EDGE:
            q.warnings.append(
                f"low resolution ({short_edge}px short edge); OCR accuracy will suffer"
            )
        return q
    blur = laplacian_variance(image)
    glare = glare_ratio(image)
    q = Quality(
        blur_score=round(blur, 2),
        glare_ratio=round(glare, 5),
        is_blurry=blur < blur_threshold,
        has_glare=glare > glare_threshold,
    )
    if q.is_blurry:
        q.warnings.append(f"blurry (laplacian variance {blur:.0f} < {blur_threshold:.0f})")
    if q.has_glare:
        q.warnings.append(f"glare on {glare:.1%} of the image")
    short_edge = min(image.shape[:2])
    if short_edge < MIN_SHORT_EDGE:
        q.warnings.append(f"low resolution ({short_edge}px short edge); OCR accuracy will suffer")
    return q
