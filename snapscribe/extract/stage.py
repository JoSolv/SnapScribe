"""Extract stage: run every configured engine over every bubble crop."""

from __future__ import annotations

from pathlib import Path

from snapscribe.core.models import Extraction, RegionKind
from snapscribe.core.pipeline import PipelineContext, StageError
from snapscribe.core.session import Session
from snapscribe.extract.base import BaseExtractor, get_extractor
from snapscribe.extract.emoji import insert_placeholders
from snapscribe.extract.preprocess import prepare_crop
from snapscribe.imaging import load_image

EXTRACTABLE = (RegionKind.BUBBLE, RegionKind.SYSTEM, RegionKind.DATE_SEPARATOR)


def build_engines(ctx: PipelineContext) -> list[BaseExtractor]:
    """Resolve the requested engines, skipping unavailable ones with a note."""
    requested = ctx.option("engines") or ["ocr"]
    engines: list[BaseExtractor] = []
    missing: list[str] = []
    for name in requested:
        if name in ("none", ""):
            continue
        kwargs: dict[str, object] = {}
        if name == "vision":
            if ctx.option("vision_model"):
                kwargs["model"] = ctx.option("vision_model")
            if ctx.option("vision_host"):
                kwargs["host"] = ctx.option("vision_host")
        try:
            engine = get_extractor(name, **kwargs)
        except KeyError as exc:
            raise StageError(str(exc)) from exc
        if not engine.available():
            missing.append(engine.describe())
            continue
        engines.append(engine)
    if not engines:
        raise StageError(
            "no extraction engine is available ("
            + ("; ".join(missing) if missing else "none requested")
            + "). Install the OCR extra with `uv pip install -e '.[ocr]'`, "
            "or pass --engine none to skip extraction."
        )
    ctx.options["engines_missing"] = missing
    return engines


class ExtractStage:
    name = "extract"
    requires = ("layout",)

    def run(self, session: Session, ctx: PipelineContext) -> None:
        profile = ctx.require_profile()
        engines = build_engines(ctx)
        notes = [f"missing engine: {m}" for m in ctx.option("engines_missing", [])]
        notes.append("engines: " + ", ".join(e.describe() for e in engines))

        targets: list[tuple[str, object]] = []
        for ref in session.images:
            for region in session.regions_for(ref.id):
                if region.kind in EXTRACTABLE:
                    targets.append((ref.id, region))
        if not targets:
            raise StageError("no regions to extract; run layout first")

        total = len(targets) * len(engines)
        done = 0
        session.extractions = {}
        errors = 0

        for ref in session.images:
            ctx.check_cancelled()
            regions = [r for r in session.regions_for(ref.id) if r.kind in EXTRACTABLE]
            if not regions:
                continue
            image = load_image(ref.path)
            for region in regions:
                crop, scale = prepare_crop(
                    image, region, blank_media=ctx.option("blank_media", True)
                )
                for engine in engines:
                    result = engine.extract(crop)
                    if engine.name == "ocr" and result.ok:
                        result = insert_placeholders(
                            result,
                            crop,
                            placeholder=profile.emoji_placeholder,
                            policy=profile.emoji_policy,
                            scale=scale,
                        )
                    if result.error:
                        errors += 1
                    session.add_extraction(
                        Extraction(
                            region_id=region.id,
                            engine=result.engine,
                            text=result.text,
                            confidence=result.confidence,
                            duration_ms=result.duration_ms,
                            lines=[line.text for line in result.lines],
                            line_confidences=[round(line.confidence, 4) for line in result.lines],
                            error=result.error,
                            meta=dict(result.meta),
                        )
                    )
                    done += 1
                    ctx.report(self.name, done / max(1, total), f"{ref.name} {region.id}")

        if ctx.debug_dir is not None:
            self._write_overlays(session, ctx)

        record = session.stage(self.name)
        record.counts = {
            "regions": len(targets),
            "extractions": sum(len(v) for v in session.extractions.values()),
            "errors": errors,
            "empty": sum(
                1
                for bucket in session.extractions.values()
                for e in bucket
                if e.engine == "ocr" and not e.text.strip()
            ),
        }
        record.notes = notes

    def _write_overlays(self, session: Session, ctx: PipelineContext) -> None:
        from snapscribe.layout.overlay import save_overlay

        debug_dir = Path(ctx.debug_dir)
        for ref in session.images:
            text = {}
            for region in session.regions_for(ref.id):
                extraction = session.extraction(region.id, "ocr") or next(
                    iter(session.extractions.get(region.id, [])), None
                )
                if extraction is not None:
                    text[region.id] = extraction.text
            save_overlay(
                debug_dir / f"{Path(ref.path).stem}_text.png",
                load_image(ref.path),
                session.regions_for(ref.id),
                text=text,
                scale=ctx.option("overlay_scale", 1.0),
            )
