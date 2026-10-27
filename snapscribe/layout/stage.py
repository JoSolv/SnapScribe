"""Layout stage: segment every image's chat area into regions."""

from __future__ import annotations

from pathlib import Path

from snapscribe.core.models import RegionKind
from snapscribe.core.pipeline import PipelineContext, StageError
from snapscribe.core.session import Session
from snapscribe.imaging import bgr_to_hex, hex_to_bgr, load_image
from snapscribe.layout.bubbles import segment_chat
from snapscribe.layout.overlay import save_overlay


class LayoutStage:
    name = "layout"
    requires = ("normalize",)

    def run(self, session: Session, ctx: PipelineContext) -> None:
        profile = ctx.require_profile()
        if not session.images:
            raise StageError("no images; run ingest first")

        debug_dir = ctx.debug_dir
        if debug_dir is not None:
            debug_dir = Path(debug_dir)
            debug_dir.mkdir(parents=True, exist_ok=True)

        notes: list[str] = []
        total_bubbles = 0
        total_dropped = 0

        for ref in session.images:
            ctx.check_cancelled()
            image = load_image(ref.path)
            chat = next(
                (r for r in session.regions_for(ref.id, RegionKind.CHAT_AREA)),
                None,
            )
            if chat is None:
                raise StageError(f"{ref.name}: no chat area; run normalize first")

            background = None
            if ref.meta.get("background"):
                background = hex_to_bgr(ref.meta["background"])
            result = segment_chat(image, chat, profile, image_id=ref.id, background=background)

            chrome_regions = [
                r
                for r in session.regions_for(ref.id)
                if r.kind in (RegionKind.CHAT_AREA, RegionKind.HEADER, RegionKind.FOOTER)
            ]
            session.regions[ref.id] = chrome_regions + result.regions
            ref.meta["background"] = bgr_to_hex(result.background)

            kept = [r for r in result.regions if r.kind is not RegionKind.DROPPED]
            dropped = [r for r in result.regions if r.kind is RegionKind.DROPPED]
            total_bubbles += len(kept)
            total_dropped += len(dropped)
            notes.extend(f"{ref.name}: {n}" for n in result.notes)

            if debug_dir is not None:
                save_overlay(
                    debug_dir / f"{Path(ref.path).stem}_layout.png",
                    image,
                    session.regions_for(ref.id),
                    scale=ctx.option("overlay_scale", 1.0),
                )
            ctx.report(
                self.name,
                (ref.index + 1) / len(session.images),
                f"{ref.name}: {len(kept)} regions",
            )

        record = session.stage(self.name)
        record.counts = {
            "bubbles": total_bubbles,
            "dropped": total_dropped,
            "system": len(session.all_regions(RegionKind.SYSTEM)),
        }
        record.notes = notes
        if total_bubbles == 0:
            raise StageError(
                "layout found no bubbles on any image. Check the debug overlay "
                "(--overlay DIR) and the profile's segmentation settings."
            )
