"""Putting images in the order the conversation was captured in.

Screenshot filenames usually carry a timestamp (``Screenshot_20240312-174501``)
and are the most reliable signal; camera photos carry EXIF instead. Both beat
mtime, which a copy or a cloud sync destroys.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from snapscribe.core.models import SourceKind

OrderMode = Literal["auto", "name", "exif", "mtime"]

#: Patterns seen on Android/iOS screenshots and common camera exports.
_FILENAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    # 20240312_174501 / 20240312-174501 / 20240312174501
    re.compile(
        r"(?P<Y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})[_\-\. ]?(?P<H>\d{2})(?P<M>\d{2})(?P<S>\d{2})?"
    ),
    # 2024-03-12 at 17.45.01 (macOS) / 2024-03-12 17:45:01
    re.compile(
        r"(?P<Y>20\d{2})-(?P<m>\d{2})-(?P<d>\d{2})\D{1,5}(?P<H>\d{2})[\.:](?P<M>\d{2})(?:[\.:](?P<S>\d{2}))?"
    ),
    # bare date
    re.compile(r"(?P<Y>20\d{2})[-_](?P<m>\d{2})[-_](?P<d>\d{2})"),
)

_NUM_RE = re.compile(r"(\d+)")


def parse_filename_datetime(name: str) -> dt.datetime | None:
    """Pull a capture timestamp out of a filename, if one is encoded there."""
    for pattern in _FILENAME_PATTERNS:
        m = pattern.search(name)
        if not m:
            continue
        g = m.groupdict()
        try:
            return dt.datetime(
                int(g["Y"]),
                int(g["m"]),
                int(g["d"]),
                int(g.get("H") or 0),
                int(g.get("M") or 0),
                int(g.get("S") or 0),
            )
        except (ValueError, TypeError):
            continue
    return None


def read_exif(path: Path) -> dict[str, object]:
    """EXIF tags as a name-keyed dict; empty if there are none."""
    try:
        from PIL import ExifTags, Image

        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return {}
            out: dict[str, object] = {}
            for tag_id, value in exif.items():
                out[ExifTags.TAGS.get(tag_id, str(tag_id))] = value
            ifd = exif.get_ifd(0x8769)  # ExifOffset
            for tag_id, value in ifd.items():
                out[ExifTags.TAGS.get(tag_id, str(tag_id))] = value
            return out
    except Exception:
        return {}


def read_exif_datetime(path: Path, exif: dict[str, object] | None = None) -> dt.datetime | None:
    tags = exif if exif is not None else read_exif(path)
    for key in ("DateTimeOriginal", "DateTimeDigitized", "DateTime"):
        raw = tags.get(key)
        if not isinstance(raw, str):
            continue
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
            try:
                return dt.datetime.strptime(raw.strip(), fmt)
            except ValueError:
                continue
    return None


def classify_source(path: Path, exif: dict[str, object] | None = None) -> SourceKind:
    """Guess whether an image is a device screenshot or a photo of a screen.

    Camera captures carry optical tags (aperture, exposure, focal length) that
    a screenshot never has; screenshots often say so in the filename.
    """
    tags = exif if exif is not None else read_exif(path)
    name = path.name.lower()
    if any(
        k in name for k in ("screenshot", "screen_shot", "skärmbild", "capture", "bildschirmfoto")
    ):
        return SourceKind.SCREENSHOT
    optical = any(
        k in tags
        for k in ("FNumber", "ExposureTime", "FocalLength", "ISOSpeedRatings", "LensModel")
    )
    if optical:
        return SourceKind.PHOTO
    if "Make" in tags or "Model" in tags:
        # Phones tag their own screenshots with Make/Model but no optics.
        return SourceKind.SCREENSHOT
    return SourceKind.UNKNOWN


def natural_key(name: str) -> tuple[object, ...]:
    """Sort key where ``img2`` comes before ``img10``."""
    parts = _NUM_RE.split(name.lower())
    return tuple(int(p) if p.isdigit() else p for p in parts)


def order_files(
    paths: Iterable[Path], mode: OrderMode = "auto"
) -> list[tuple[Path, dt.datetime | None, str]]:
    """Order images for transcription.

    Returns ``(path, taken_at, order_source)`` triples. In ``auto`` mode a
    single source is chosen for the *whole batch* — mixing filename timestamps
    for some images and mtime for others produces nonsense orderings — and the
    first source that covers every file wins.
    """
    items = list(paths)
    if not items:
        return []

    by_name = {p: parse_filename_datetime(p.name) for p in items}
    by_exif = {p: read_exif_datetime(p) for p in items}
    by_mtime = {p: dt.datetime.fromtimestamp(p.stat().st_mtime) for p in items}

    if mode == "name":
        chosen, source = by_name, "filename"
    elif mode == "exif":
        chosen, source = by_exif, "exif"
    elif mode == "mtime":
        chosen, source = by_mtime, "mtime"
    else:
        candidates = (("filename", by_name), ("exif", by_exif), ("mtime", by_mtime))
        chosen, source = by_mtime, "mtime"
        for name, table in candidates:
            values = [v for v in table.values() if v is not None]
            if len(values) == len(items) and len(set(values)) > 1:
                chosen, source = table, name
                break
            if len(values) == len(items) and len(items) == 1:
                chosen, source = table, name
                break

    def key(p: Path) -> tuple[object, ...]:
        stamp = chosen.get(p)
        return (
            0 if stamp is not None else 1,
            stamp or dt.datetime.min,
            *natural_key(p.name),
        )

    ordered = sorted(items, key=key)
    return [
        (p, chosen.get(p) or by_mtime.get(p), source if chosen.get(p) else "name") for p in ordered
    ]
