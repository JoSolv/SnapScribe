"""Turning user input (folders, zips, globs, files) into a list of images."""

from __future__ import annotations

import glob
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from snapscribe.imaging import IMAGE_EXTENSIONS, register_heif

MAX_ZIP_ENTRIES = 5_000
MAX_ZIP_BYTES = 4 << 30  # 4 GiB uncompressed, guards against zip bombs


@dataclass
class ImportedFile:
    """An image found on disk, plus where it came from."""

    path: Path
    origin: str
    """The input the user named (a folder, a zip, or the file itself)."""
    archive_member: str = ""
    warnings: list[str] = field(default_factory=list)


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTENSIONS


def _hidden(p: Path, root: Path) -> bool:
    try:
        rel = p.relative_to(root)
    except ValueError:
        rel = Path(p.name)
    return any(part.startswith(".") or part == "__MACOSX" for part in rel.parts)


def discover_images(root: Path, recursive: bool = True) -> list[Path]:
    """All image files under ``root``, skipping hidden and macOS cruft."""
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in root.glob(pattern) if p.is_file() and _is_image(p) and not _hidden(p, root)
    )


def extract_zip(archive: Path, dest: Path) -> list[ImportedFile]:
    """Extract the image members of ``archive`` into ``dest``.

    Member names are flattened and sanitised, so a crafted zip cannot write
    outside ``dest`` (path traversal) — ``zipfile.extract`` sanitises too, but
    being explicit keeps the ordering keys (the member names) readable.
    """
    dest.mkdir(parents=True, exist_ok=True)
    out: list[ImportedFile] = []
    with zipfile.ZipFile(archive) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_ZIP_ENTRIES:
            raise ValueError(
                f"{archive.name}: {len(infos)} entries exceeds the {MAX_ZIP_ENTRIES} limit"
            )
        total = sum(i.file_size for i in infos)
        if total > MAX_ZIP_BYTES:
            raise ValueError(f"{archive.name}: {total} uncompressed bytes exceeds the limit")
        for info in infos:
            member = Path(info.filename)
            if not _is_image(member):
                continue
            if any(
                part in ("..", "") or part.startswith(".") or part == "__MACOSX"
                for part in member.parts
            ):
                continue
            flat = "_".join(member.parts)
            target = dest / flat
            counter = 1
            while target.exists():
                target = dest / f"{Path(flat).stem}_{counter}{Path(flat).suffix}"
                counter += 1
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out.append(ImportedFile(path=target, origin=str(archive), archive_member=info.filename))
    return sorted(out, key=lambda f: f.archive_member)


def expand_inputs(
    inputs: list[str | Path],
    workdir: Path | None = None,
    recursive: bool = True,
) -> tuple[list[ImportedFile], list[str]]:
    """Resolve the user's inputs into concrete image files.

    Returns the files plus any warnings (missing paths, unreadable zips,
    unsupported extensions that looked like they were meant to be images).
    """
    register_heif()
    warnings: list[str] = []
    found: list[ImportedFile] = []
    extracted_root = workdir / "extracted" if workdir else None

    for raw in inputs:
        p = Path(raw).expanduser()
        if not p.exists():
            # Shells on Windows do not expand globs, and a quoted pattern
            # reaches us verbatim anywhere. `glob` handles absolute patterns,
            # which `Path.glob` refuses.
            matches = sorted(Path(m) for m in glob.glob(str(raw)))
            if not matches:
                warnings.append(f"input not found: {raw}")
                continue
            for m in matches:
                found.extend(expand_inputs([m], workdir, recursive)[0])
            continue
        if p.is_dir():
            images = discover_images(p, recursive=recursive)
            if not images:
                warnings.append(f"no images in folder: {p}")
            found.extend(ImportedFile(path=i, origin=str(p)) for i in images)
        elif p.suffix.lower() == ".zip":
            dest = (extracted_root or Path(tempfile.mkdtemp(prefix="snapscribe-zip-"))) / p.stem
            try:
                members = extract_zip(p, dest)
            except (zipfile.BadZipFile, ValueError) as exc:
                warnings.append(f"could not read zip {p}: {exc}")
                continue
            if not members:
                warnings.append(f"no images in zip: {p}")
            found.extend(members)
        elif _is_image(p):
            found.append(ImportedFile(path=p, origin=str(p)))
        else:
            warnings.append(f"unsupported file type, skipped: {p}")

    # De-duplicate by resolved path, keeping first occurrence.
    seen: set[Path] = set()
    unique: list[ImportedFile] = []
    for f in found:
        key = f.path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(f)
    return unique, warnings
