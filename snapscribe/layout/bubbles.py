"""Bubble segmentation.

The approach, in order:

1. Measure the chat background (the modal colour of the chat area).
2. Mask every pixel that is far enough from it — bubbles, text, avatars, all of
   it. This works because chat UIs are flat-coloured by design.
3. Close the mask horizontally so a bubble's text lines merge into one blob
   (needed when the bubble fill *is* the background, as in some minimal apps),
   then open it to drop single-pixel noise.
4. Take connected components, split ones that swallowed two stacked bubbles,
   and filter by size.
5. Classify each: side from position, with bubble colour as the second signal;
   narrow centred blobs are system messages or date separators.
6. Drop bubbles touching the top or bottom of the chat area — they are clipped
   and the same message will be whole in the neighbouring screenshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from snapscribe.core.models import Region, RegionKind, Side
from snapscribe.imaging import color_distance, dominant_color
from snapscribe.profiles import Profile


@dataclass
class SegmentationResult:
    regions: list[Region] = field(default_factory=list)
    background: tuple[int, int, int] = (0, 0, 0)
    mask: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def bubbles(self) -> list[Region]:
        return [r for r in self.regions if r.kind in (RegionKind.BUBBLE, RegionKind.SYSTEM)]


def content_mask(chat: np.ndarray, background: tuple[int, int, int], tolerance: int) -> np.ndarray:
    """Pixels that differ from the background on any channel."""
    diff = np.abs(chat.astype(np.int16) - np.asarray(background, dtype=np.int16)).max(axis=2)
    return (diff > tolerance).astype(np.uint8)


def _morph(
    mask: np.ndarray, close: tuple[int, int], open_: tuple[int, int], dilate: int
) -> np.ndarray:
    out = mask
    if open_[0] > 1 or open_[1] > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones(open_[::-1], np.uint8))
    if close[0] > 1 or close[1] > 1:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones(close[::-1], np.uint8))
    if dilate:
        out = cv2.dilate(out, np.ones((dilate, dilate), np.uint8))
    return out


def _split_stacked(
    mask: np.ndarray, box: tuple[int, int, int, int], row_gap: int, min_height: int
) -> list[tuple[int, int, int, int]]:
    """Split a component that merged two vertically stacked bubbles.

    Uses the *unclosed* mask so the real gap between bubbles is still visible.
    """
    x, y, w, h = box
    band = mask[y : y + h, x : x + w]
    coverage = band.sum(axis=1) / max(1, w)
    gaps: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(coverage):
        if value <= 0.005:
            if start is None:
                start = i
        elif start is not None:
            if i - start >= row_gap:
                gaps.append((start, i))
            start = None
    if start is not None and len(coverage) - start >= row_gap:
        gaps.append((start, len(coverage)))

    cuts = [g for g in gaps if g[0] > 0 and g[1] < h]
    if not cuts:
        return [box]
    pieces: list[tuple[int, int, int, int]] = []
    prev = 0
    for g0, g1 in [*cuts, (h, h)]:
        piece_h = g0 - prev
        if piece_h >= min_height:
            sub = band[prev:g0]
            cols = np.flatnonzero(sub.any(axis=0))
            rows = np.flatnonzero(sub.any(axis=1))
            if cols.size and rows.size:
                pieces.append(
                    (
                        x + int(cols[0]),
                        y + prev + int(rows[0]),
                        int(cols[-1] - cols[0]) + 1,
                        int(rows[-1] - rows[0]) + 1,
                    )
                )
        prev = g1
    return pieces or [box]


def remove_icon_blobs(
    mask: np.ndarray, max_size: int, min_size: int = 20
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Strip avatars and round icons out of the mask before closing.

    A group-chat avatar sits a few pixels from its bubble, and the horizontal
    close below would weld the two into one component with a box that is wrong
    on the left edge. Avatars are small, square and *solid*; glyphs of the same
    size are neither (a letter fills well under half its bounding box), so the
    fill ratio keeps this from eating text in apps whose bubbles have no fill.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = mask.copy()
    removed: list[tuple[int, int, int, int]] = []
    for i in range(1, count):
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (min_size <= w <= max_size and min_size <= h <= max_size):
            continue
        aspect = w / h
        if not 0.7 <= aspect <= 1.43:
            continue
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area / max(1, w * h) < 0.6:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        out[labels == i] = 0
        removed.append((x, y, w, h))
    return out, removed


def _component_fill(chat: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
    """The bubble's fill colour: the modal colour of its interior."""
    x, y, w, h = box
    inset_x = min(max(2, w // 8), max(2, w // 2 - 1))
    inset_y = min(max(2, h // 8), max(2, h // 2 - 1))
    interior = chat[y + inset_y : y + h - inset_y, x + inset_x : x + w - inset_x]
    if interior.size == 0:
        interior = chat[y : y + h, x : x + w]
    return dominant_color(interior)


def _row_color_counts(crop: np.ndarray, bits: int = 4) -> np.ndarray:
    """Distinct quantised colours per row — cheap "is this a photo?" signal."""
    shift = 8 - bits
    q = (crop >> shift).astype(np.uint32)
    keys = (q[:, :, 0] << (2 * bits)) | (q[:, :, 1] << bits) | q[:, :, 2]
    return np.array([np.unique(row).size for row in keys], dtype=np.int32)


def _column_color_counts(crop: np.ndarray, bits: int = 4) -> np.ndarray:
    shift = 8 - bits
    q = (crop >> shift).astype(np.uint32)
    keys = (q[:, :, 0] << (2 * bits)) | (q[:, :, 1] << bits) | q[:, :, 2]
    return np.array([np.unique(keys[:, c]).size for c in range(keys.shape[1])], dtype=np.int32)


def media_rows(
    crop: np.ndarray,
    color_count: int,
    min_rows: int,
    min_width_fraction: float = 0.5,
) -> list[tuple[int, int]]:
    """Row spans inside a bubble that look like an image rather than text.

    Colour richness alone would also flag a bubble containing nothing but
    emoji, which *is* text as far as transcription is concerned. A photo is
    rich across most of the bubble's width; a row of emoji is rich in a narrow
    band, so each candidate span must also be wide.
    """
    counts = _row_color_counts(crop)
    rich = counts > color_count
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(rich):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start >= min_rows:
                spans.append((start, i))
            start = None
    if start is not None and len(rich) - start >= min_rows:
        spans.append((start, len(rich)))

    width = max(1, crop.shape[1])
    wide: list[tuple[int, int]] = []
    for a, b in spans:
        col_counts = _column_color_counts(crop[a:b])
        rich_cols = int((col_counts > max(3, color_count // 2)).sum())
        if rich_cols / width >= min_width_fraction:
            wide.append((a, b))
    return wide


def _edge_is_flat(band: np.ndarray, row: int, threshold: float = 0.95) -> bool:
    """True when ``row`` of a bubble's mask is as wide as the bubble itself.

    A bubble with rounded corners tapers at its first and last rows. A flat
    edge means the viewport cut the bubble off there.
    """
    widths = band.sum(axis=1)
    peak = int(widths.max())
    if peak <= 0:
        return False
    return int(widths[row]) / peak >= threshold


def detect_truncation(
    mask: np.ndarray,
    box: tuple[int, int, int, int],
    chat_height: int,
    profile: Profile,
) -> str:
    """Return "top"/"bottom"/"" for a bubble the viewport cut through.

    The geometric test (touching the chat-area edge) misses the common case
    where the app's composer background is the same colour as the chat
    background: chrome detection then stops a few pixels short, and a cut
    bubble ends just *inside* the chat area. The flat-edge test catches those.
    """
    x, y, w, h = box
    edge = profile.layout.edge_tolerance_px
    if y <= edge:
        return "top"
    if (y + h) >= (chat_height - edge):
        return "bottom"
    if not profile.layout.rounded_bubbles:
        return ""
    zone = max(edge, int(chat_height * profile.layout.clip_zone_fraction))
    band = mask[y : y + h, x : x + w]
    if band.size == 0:
        return ""
    if y <= zone and _edge_is_flat(band, 0):
        return "top"
    if (y + h) >= (chat_height - zone) and _edge_is_flat(band, band.shape[0] - 1):
        return "bottom"
    return ""


def classify_side(
    region_box: tuple[int, int, int, int],
    chat_width: int,
    fill: tuple[int, int, int],
    profile: Profile,
) -> tuple[Side, float, str]:
    """Left or right, from position first and colour second."""
    x, _, w, _ = region_box
    centre = (x + w / 2) / max(1, chat_width)
    split = profile.layout.side_split
    dead = profile.layout.side_deadzone

    by_position: Side | None
    if centre < split - dead:
        by_position = Side.LEFT
    elif centre > split + dead:
        by_position = Side.RIGHT
    else:
        by_position = None

    incoming = profile.colors.bgr("incoming_bubble")
    outgoing = profile.colors.bgr("outgoing_bubble")
    by_color: Side | None = None
    colour_margin = 0.0
    if incoming is not None and outgoing is not None:
        d_in = color_distance(fill, incoming)
        d_out = color_distance(fill, outgoing)
        tol = profile.colors.tolerance
        if min(d_in, d_out) <= tol:
            by_color = Side.LEFT if d_in < d_out else Side.RIGHT
            colour_margin = abs(d_in - d_out)

    if by_position is not None and by_color is not None:
        if by_position == by_color:
            return by_position, 1.0, "position+colour"
        # Position is usually right; colour wins only when it is unambiguous.
        if colour_margin > profile.colors.tolerance:
            return by_color, 0.55, "colour (disagrees with position)"
        return by_position, 0.55, "position (disagrees with colour)"
    if by_position is not None:
        return by_position, 0.85, "position"
    if by_color is not None:
        return by_color, 0.7, "colour"
    return Side.UNKNOWN, 0.3, "undecided"


def _drop_nested(regions: list[Region]) -> None:
    """Drop regions that sit entirely inside another one.

    A bubble whose inner card or quoted-reply block is close in colour to the
    chat background leaves a hole in the mask, and whatever is drawn inside
    that hole becomes its own component. It is never a separate message.
    """
    candidates = [r for r in regions if r.kind in (RegionKind.BUBBLE, RegionKind.SYSTEM)]
    for inner in candidates:
        for outer in candidates:
            if inner is outer or outer.kind is RegionKind.DROPPED:
                continue
            if outer.area <= inner.area:
                continue
            ix0, iy0, ix1, iy1 = inner.box
            ox0, oy0, ox1, oy1 = outer.box
            if ox0 <= ix0 and oy0 <= iy0 and ox1 >= ix1 and oy1 >= iy1:
                inner.kind = RegionKind.DROPPED
                inner.drop_reason = f"nested inside {outer.id}"
                break


def segment_chat(
    image: np.ndarray,
    chat: Region,
    profile: Profile,
    image_id: str = "",
    background: tuple[int, int, int] | None = None,
) -> SegmentationResult:
    """Find every message region inside one screenshot's chat area."""
    seg = profile.segmentation
    image_id = image_id or chat.image_id
    crop = image[chat.y : chat.y + chat.h, chat.x : chat.x + chat.w]
    if crop.size == 0:
        return SegmentationResult(notes=["empty chat area"])

    bg = background or dominant_color(crop)
    raw = content_mask(crop, bg, seg.bg_tolerance)
    icon_limit = int(chat.w * max(profile.layout.avatar_width_fraction, 0.085))
    raw, icons = remove_icon_blobs(raw, icon_limit)
    mask = _morph(raw, seg.close_kernel, seg.open_kernel, seg.dilate_px)

    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    chat_w, chat_h = chat.w, chat.h
    min_w = int(chat_w * seg.min_width_fraction)
    max_w = int(chat_w * seg.max_width_fraction)
    max_h = int(chat_h * seg.max_height_fraction)

    boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < seg.min_area_px:
            continue
        if w > max_w and h > max_h:
            continue
        boxes.extend(_split_stacked(raw, (x, y, w, h), seg.row_gap_px, seg.min_height_px))

    regions: list[Region] = [
        Region(
            id=f"{image_id}:icon{i:02d}",
            image_id=image_id,
            kind=RegionKind.AVATAR,
            x=chat.x + ix,
            y=chat.y + iy,
            w=iw,
            h=ih,
            parent_id=chat.id,
            side=Side.UNKNOWN,
            confidence=0.5,
        )
        for i, (ix, iy, iw, ih) in enumerate(icons)
    ]
    notes: list[str] = []

    for n, (x, y, w, h) in enumerate(sorted(boxes, key=lambda b: (b[1], b[0]))):
        region = Region(
            id=f"{image_id}:r{n:03d}",
            image_id=image_id,
            kind=RegionKind.BUBBLE,
            x=chat.x + x,
            y=chat.y + y,
            w=w,
            h=h,
            parent_id=chat.id,
            order=n,
        )
        region.meta["chat_x"] = x
        region.meta["chat_y"] = y

        if w < min_w or h < seg.min_height_px:
            region.kind = RegionKind.DROPPED
            region.drop_reason = f"too small ({w}x{h}, minimum {min_w}x{seg.min_height_px})"
            regions.append(region)
            continue
        if w > max_w:
            region.kind = RegionKind.DROPPED
            region.drop_reason = f"wider than a bubble ({w} > {max_w})"
            regions.append(region)
            continue
        if h > max_h:
            region.kind = RegionKind.DROPPED
            region.drop_reason = f"taller than a bubble ({h} > {max_h})"
            regions.append(region)
            continue
        truncated = detect_truncation(raw, (x, y, w, h), chat_h, profile)
        if truncated:
            region.kind = RegionKind.DROPPED
            region.drop_reason = f"clipped at the {truncated} of the chat area"
            regions.append(region)
            continue

        fill = _component_fill(crop, (x, y, w, h))
        region.fill_color = fill

        centre = (x + w / 2) / max(1, chat_w)
        is_narrow = w <= chat_w * seg.system_max_width_fraction
        is_centred = abs(centre - 0.5) <= seg.system_center_tolerance
        if is_narrow and is_centred:
            region.kind = RegionKind.SYSTEM
            region.side = Side.CENTER
            region.confidence = 0.8
            region.meta["side_basis"] = "centred"
        else:
            side, confidence, basis = classify_side((x, y, w, h), chat_w, fill, profile)
            region.side = side
            region.confidence = confidence
            region.meta["side_basis"] = basis

        bubble_crop = crop[y : y + h, x : x + w]
        spans = media_rows(bubble_crop, seg.attachment_color_count, seg.attachment_min_rows)
        if spans:
            covered = sum(b - a for a, b in spans)
            region.meta["media_rows"] = [[int(a), int(b)] for a, b in spans]
            region.meta["media_coverage"] = round(covered / max(1, h), 3)
            region.meta["attachment"] = True
        regions.append(region)

    _drop_nested(regions)

    kept = [r for r in regions if r.kind in (RegionKind.BUBBLE, RegionKind.SYSTEM)]
    for order, region in enumerate(sorted(kept, key=lambda r: (r.y, r.x))):
        region.order = order
    if not kept:
        notes.append(
            "no bubbles found; try a lower segmentation.bg_tolerance or check the profile colours"
        )
    return SegmentationResult(regions=regions, background=bg, mask=mask, notes=notes)
