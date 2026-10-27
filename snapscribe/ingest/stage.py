"""The ingest stage: inputs in, ordered ``ImageRef`` list out."""

from __future__ import annotations

from pathlib import Path

from snapscribe.core.models import ImageRef
from snapscribe.core.pipeline import PipelineContext, StageError
from snapscribe.core.session import Session
from snapscribe.imaging import file_sha1, register_heif
from snapscribe.ingest.loader import expand_inputs
from snapscribe.ingest.ordering import classify_source, order_files, read_exif


def _dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


def ingest(
    session: Session,
    inputs: list[str | Path],
    order: str = "auto",
    recursive: bool = True,
    hash_files: bool = True,
) -> list[str]:
    """Populate ``session.images``. Returns warnings."""
    register_heif()
    workdir = Path(session.workdir) if session.workdir else None
    files, warnings = expand_inputs(inputs, workdir=workdir, recursive=recursive)
    if not files:
        raise StageError(
            "no images found. " + ("; ".join(warnings) if warnings else "check the input path")
        )

    origins = {f.path: f for f in files}
    ordered = order_files([f.path for f in files], mode=order)  # type: ignore[arg-type]

    session.images = []
    seen_hashes: dict[str, str] = {}
    for i, (path, taken_at, order_source) in enumerate(ordered):
        exif = read_exif(path)
        width, height = _dimensions(path)
        sha1 = file_sha1(path) if hash_files else ""
        if sha1 and sha1 in seen_hashes:
            warnings.append(f"{path.name} is byte-identical to {seen_hashes[sha1]}, kept both")
        elif sha1:
            seen_hashes[sha1] = path.name
        ref = ImageRef(
            id=f"img{i:04d}",
            path=str(path),
            index=i,
            width=width,
            height=height,
            taken_at=taken_at,
            order_source=order_source,
            source_kind=classify_source(path, exif),
            sha1=sha1,
        )
        origin = origins.get(path)
        if origin is not None and origin.archive_member:
            ref.meta["archive_member"] = origin.archive_member
        if origin is not None:
            ref.meta["origin"] = origin.origin
        session.images.append(ref)

    session.inputs = [str(i) for i in inputs]
    sizes = {(r.width, r.height) for r in session.images if r.width}
    if len(sizes) > 1:
        warnings.append(
            f"mixed image sizes {sorted(sizes)}; header/footer detection falls back to "
            "per-size groups"
        )
    return warnings


class IngestStage:
    name = "ingest"
    requires: tuple[str, ...] = ()

    def run(self, session: Session, ctx: PipelineContext) -> None:
        inputs = ctx.option("inputs") or session.inputs
        if not inputs:
            raise StageError("ingest needs at least one input path")
        warnings = ingest(
            session,
            list(inputs),
            order=ctx.option("order", "auto"),
            recursive=ctx.option("recursive", True),
            hash_files=ctx.option("hash_files", True),
        )
        record = session.stage(self.name)
        record.counts = {"images": len(session.images)}
        record.notes = warnings
        for i, image in enumerate(session.images, 1):
            ctx.report(self.name, i / max(1, len(session.images)), image.name)
