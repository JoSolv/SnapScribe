"""Shared image I/O and colour helpers.

Everything downstream works on uint8 BGR numpy arrays (OpenCV convention), so
all loading funnels through :func:`load_image`.
"""

from __future__ import annotations

import contextlib
import hashlib
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".avif"}
)

_heif_registered = False


def register_heif() -> bool:
    """Teach Pillow about HEIC/HEIF/AVIF. Returns False if unavailable."""
    global _heif_registered
    if _heif_registered:
        return True
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
        with contextlib.suppress(AttributeError):  # older pillow-heif has no AVIF
            pillow_heif.register_avif_opener()
        _heif_registered = True
    except Exception:
        return False
    return True


def load_image(path: str | Path) -> np.ndarray:
    """Load any supported image as uint8 BGR.

    OpenCV cannot read HEIC and is picky about non-ASCII paths, so Pillow is
    the loader and OpenCV only ever sees arrays.
    """
    from PIL import Image, ImageOps

    p = Path(path)
    if p.suffix.lower() in {".heic", ".heif", ".avif"}:
        register_heif()
    with Image.open(p) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        arr = np.asarray(im)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def save_image(path: str | Path, image: np.ndarray) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", image)
    if not ok:
        raise OSError(f"could not encode {p}")
    p.write_bytes(buf.tobytes())
    return p


def file_sha1(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1(usedforsecurity=False)
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------- colours


def hex_to_bgr(value: str) -> tuple[int, int, int]:
    """``'#RRGGBB'`` (or ``'RRGGBB'``, or ``'#RGB'``) -> BGR tuple."""
    s = value.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"not a hex colour: {value!r}")
    r, g, b = (int(s[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def bgr_to_hex(color: tuple[int, int, int] | np.ndarray) -> str:
    b, g, r = (int(c) for c in color[:3])
    return f"#{r:02X}{g:02X}{b:02X}"


def color_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    """Euclidean distance in BGR space (good enough for flat UI fills)."""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def dominant_color(image: np.ndarray, bins: int = 16) -> tuple[int, int, int]:
    """Modal colour of an image, found on a coarse grid then refined.

    Quantising first stops anti-aliasing and gradients from splitting the
    background into thousands of near-identical colours.
    """
    if image.size == 0:
        return (0, 0, 0)
    flat = image.reshape(-1, 3)
    step = max(1, flat.shape[0] // 200_000)
    sample = flat[::step]
    q = (sample.astype(np.uint16) // bins).astype(np.uint8)
    keys = q[:, 0].astype(np.uint32) * 65536 + q[:, 1].astype(np.uint32) * 256 + q[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    top = values[int(np.argmax(counts))]
    centre = np.array([top // 65536, (top // 256) % 256, top % 256], dtype=np.uint16) * bins
    lo = centre.astype(np.int16)
    hi = lo + bins
    mask = np.all((sample >= lo) & (sample < hi), axis=1)
    if not mask.any():
        return tuple(int(c) for c in centre)  # type: ignore[return-value]
    return tuple(int(c) for c in np.median(sample[mask], axis=0))  # type: ignore[return-value]


def is_dark(color: tuple[int, int, int]) -> bool:
    b, g, r = (float(c) for c in color[:3])
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    return luma < 128.0


def laplacian_variance(image: np.ndarray) -> float:
    """Blur metric: variance of the Laplacian. Higher is sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def glare_ratio(image: np.ndarray, threshold: int = 250, min_blob_frac: float = 0.0005) -> float:
    """Fraction of pixels in large near-saturated blobs.

    Individual blown-out pixels are everywhere in normal images; glare shows up
    as *contiguous* saturated areas, so small blobs are discarded.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mask = (gray >= threshold).astype(np.uint8)
    if not mask.any():
        return 0.0
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = gray.size
    min_area = max(16, int(total * min_blob_frac))
    big = sum(
        int(stats[i, cv2.CC_STAT_AREA])
        for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    )
    return big / total
