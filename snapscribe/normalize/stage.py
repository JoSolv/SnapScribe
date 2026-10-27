"""Normalize stage: chrome detection, quality checks, profile resolution."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from snapscribe.core.models import ChromeInfo, RegionKind, Side, SourceKind
from snapscribe.core.pipeline import PipelineContext, StageError
from snapscribe.core.session import Session
from snapscribe.imaging import bgr_to_hex, load_image
from snapscribe.normalize.chrome import (
    ChromeDetection,
    background_color,
    detect_chrome,
    detect_chrome_group,
    refine_with_profile,
)
from snapscribe.normalize.header import app_bar_text, parse_header_title
from snapscribe.normalize.quality import assess_quality
from snapscribe.profiles import Profile, detect_profile, list_profiles, load_profile
from snapscribe.profiles.loader import ProfileError


def _resolve_requested_profile(ctx: PipelineContext) -> Profile | None:
    request = (ctx.profile_request or "auto").strip()
    if ctx.profile is not None:
        return ctx.profile
    if request in ("", "auto"):
        return None
    return load_profile(request)


def _read_band_lines(image: np.ndarray, top: int, bottom: int) -> list[tuple[str, int]]:
    """OCR a band, returning ``(text, y)`` per line in band coordinates."""
    if bottom - top < 10:
        return []
    from snapscribe.extract import get_extractor

    try:
        engine = get_extractor("ocr")
    except Exception:
        return []
    if not engine.available():
        return []
    try:
        result = engine.extract(image[top:bottom])
    except Exception:
        return []
    return [(line.text, line.box[1]) for line in result.lines if line.text.strip()]


def _read_band(image: np.ndarray, top: int, bottom: int) -> str:
    """OCR a horizontal band. Returns "" when OCR is unavailable.

    Used for the composer strip (the strongest app fingerprint there is) and
    the title bar (which names the conversation).
    """
    if bottom - top < 10:
        return ""
    from snapscribe.extract import get_extractor

    try:
        engine = get_extractor("ocr")
    except Exception:
        return ""
    if not engine.available():
        return ""
    try:
        return engine.extract(image[top:bottom]).text
    except Exception:
        return ""


class NormalizeStage:
    name = "normalize"
    requires = ("ingest",)

    def run(self, session: Session, ctx: PipelineContext) -> None:
        if not session.images:
            raise StageError("no images to normalize; run ingest first")

        profile = _resolve_requested_profile(ctx)
        notes: list[str] = []

        groups: dict[tuple[int, int], list[str]] = defaultdict(list)
        cache: dict[str, np.ndarray] = {}
        for ref in session.images:
            image = load_image(ref.path)
            cache[ref.id] = image
            ref.height, ref.width = image.shape[:2]
            ref.quality = assess_quality(image)
            if ref.quality.warnings and ref.source_kind is not SourceKind.PHOTO:
                notes.extend(f"{ref.name}: {w}" for w in ref.quality.warnings)
            groups[(ref.width, ref.height)].append(ref.id)

        detections: dict[str, ChromeDetection] = {}
        for size, ids in groups.items():
            images = [cache[i] for i in ids]
            bg_tol = profile.segmentation.bg_tolerance if profile else 18
            if len(images) >= 2:
                detection = detect_chrome_group(
                    images,
                    pixel_tolerance=ctx.option("pixel_tolerance", 12),
                    row_agreement=ctx.option("row_agreement", 0.85),
                    bg_tolerance=bg_tol,
                )
            else:
                detection = ChromeDetection(
                    method="none", notes=["single image in this size group"]
                )
            if detection.method != "stable_rows":
                fallback_profile = profile or load_profile("generic")
                detection = detect_chrome(
                    images[0],
                    fallback_profile.layout.header_fraction,
                    fallback_profile.layout.footer_fraction,
                )
                notes.append(
                    f"{size[0]}x{size[1]}: using profile fractions for chrome "
                    f"({'; '.join(detection.notes)})"
                )
            elif profile is not None:
                detection = refine_with_profile(
                    detection,
                    images[0],
                    profile.layout.header_fraction,
                    profile.layout.footer_fraction,
                )
            for image_id in ids:
                detections[image_id] = detection

        # Profile auto-detection runs on the measured chat area of the first
        # image, which is exactly the region the fingerprint describes.
        if profile is None:
            first = session.images[0]
            detection = detections[first.id]
            image = cache[first.id]
            chat = image[detection.header_height : image.shape[0] - detection.footer_height]
            footer_text = ""
            if ctx.option("fingerprint_ocr", True):
                footer_text = _read_band(
                    image, image.shape[0] - detection.footer_height, image.shape[0]
                )
            candidates = list_profiles()
            profile, match, ranked = detect_profile(chat, candidates, footer_text=footer_text)
            session.profile_name = profile.name
            session.profile_source = "auto"
            session.profile_match = {
                "chosen": profile.name,
                "footer_text": footer_text,
                "best": match.as_dict(),
                "ranking": [m.as_dict() for m in ranked[:5]],
            }
            notes.append(
                f"auto-detected profile {profile.name!r} (score {match.score:.2f})"
                + (f" from footer text {footer_text[:40]!r}" if footer_text else "")
            )
            # Chrome fallbacks were computed before the profile was known; redo
            # the ones that used generic fractions.
            for ids in groups.values():
                det = detections[ids[0]]
                if det.method == "profile":
                    redone = detect_chrome(
                        cache[ids[0]],
                        profile.layout.header_fraction,
                        profile.layout.footer_fraction,
                    )
                    for image_id in ids:
                        detections[image_id] = redone
        else:
            session.profile_name = profile.name
            session.profile_source = session.profile_source or "explicit"

        overrides = ctx.option("profile_overrides") or {}
        if overrides:
            from snapscribe.profiles import apply_overrides

            try:
                profile = apply_overrides(profile, overrides)
            except ProfileError as exc:
                raise StageError(str(exc)) from exc
            session.profile_source = f"{session.profile_source}+overrides"
        ctx.profile = profile

        if ctx.option("read_header", True) and not session.settings.get("conversation"):
            first = session.images[0]
            detection = detections[first.id]
            lines = _read_band_lines(cache[first.id], 0, detection.header_height)
            title = parse_header_title(app_bar_text(lines, detection.header_height))
            if title:
                session.settings["conversation"] = title
                notes.append(f"conversation title read from the header: {title!r}")

        session.chrome = {}
        session.regions = {}
        for ref in session.images:
            detection = detections[ref.id]
            image = cache[ref.id]
            info = ChromeInfo(
                image_id=ref.id,
                header_height=detection.header_height,
                footer_height=detection.footer_height,
                left_margin=round(ref.width * profile.layout.left_margin_fraction),
                right_margin=round(ref.width * profile.layout.right_margin_fraction),
                method=detection.method,
                confidence=detection.confidence,
            )
            session.chrome[ref.id] = info
            chat = info.chat_area(ref.width, ref.height)
            if chat.w <= 0 or chat.h <= 0:
                raise StageError(
                    f"{ref.name}: chrome detection left no chat area "
                    f"(header {info.header_height}px, footer {info.footer_height}px)"
                )
            session.regions[ref.id] = [chat]
            bg = background_color(image, detection)
            ref.meta["background"] = bgr_to_hex(bg)
            ref.meta["chrome_method"] = detection.method
            if info.header_height:
                session.regions[ref.id].append(
                    _band(ref.id, RegionKind.HEADER, 0, 0, ref.width, info.header_height)
                )
            if info.footer_height:
                session.regions[ref.id].append(
                    _band(
                        ref.id,
                        RegionKind.FOOTER,
                        0,
                        ref.height - info.footer_height,
                        ref.width,
                        info.footer_height,
                    )
                )
            ctx.report(self.name, (ref.index + 1) / len(session.images), ref.name)

        record = session.stage(self.name)
        record.counts = {
            "images": len(session.images),
            "blurry": sum(1 for r in session.images if r.quality.is_blurry),
            "glare": sum(1 for r in session.images if r.quality.has_glare),
        }
        record.notes = notes


def _band(image_id: str, kind: RegionKind, x: int, y: int, w: int, h: int):
    from snapscribe.core.models import Region

    return Region(
        id=f"{image_id}:{kind.value}",
        image_id=image_id,
        kind=kind,
        x=x,
        y=y,
        w=w,
        h=h,
        side=Side.CENTER,
        confidence=1.0,
    )
