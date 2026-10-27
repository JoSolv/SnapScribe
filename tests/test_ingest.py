"""Discovery, ordering and the zip path."""

from __future__ import annotations

import datetime as dt
import zipfile

import numpy as np
import pytest

from snapscribe.core.session import Session
from snapscribe.imaging import save_image
from snapscribe.ingest import expand_inputs, order_files, parse_filename_datetime
from snapscribe.ingest.stage import ingest


def _write(path, value=40):
    save_image(path, np.full((20, 10, 3), value, np.uint8))
    return path


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Screenshot_20240312-174501.png", dt.datetime(2024, 3, 12, 17, 45, 1)),
        ("Screenshot_20240312_174501_WhatsApp.jpg", dt.datetime(2024, 3, 12, 17, 45, 1)),
        ("Screenshot 2024-03-12 at 17.45.01.png", dt.datetime(2024, 3, 12, 17, 45, 1)),
        ("photo_2024-03-12.heic", dt.datetime(2024, 3, 12, 0, 0)),
        ("IMG_0001.png", None),
    ],
)
def test_parse_filename_datetime(name, expected):
    assert parse_filename_datetime(name) == expected


def test_order_prefers_filename_timestamps(tmp_path):
    _write(tmp_path / "Screenshot_20240312-174501.png")
    _write(tmp_path / "Screenshot_20240312-170000.png")
    _write(tmp_path / "Screenshot_20240312-180000.png")
    ordered = order_files(sorted(tmp_path.glob("*.png")))
    assert [p.name[-10:-4] for p, _, _ in ordered] == ["170000", "174501", "180000"]
    assert {source for _, _, source in ordered} == {"filename"}


def test_order_falls_back_to_natural_sort(tmp_path):
    for name in ("shot2.png", "shot10.png", "shot1.png"):
        _write(tmp_path / name)
    ordered = order_files(sorted(tmp_path.glob("*.png")), mode="name")
    assert [p.name for p, _, _ in ordered] == ["shot1.png", "shot2.png", "shot10.png"]


def test_expand_inputs_handles_folders_zips_and_junk(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    _write(folder / "a.png")
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    (folder / ".hidden.png").write_bytes(b"")

    inner = tmp_path / "inner"
    inner.mkdir()
    _write(inner / "b.png")
    archive = tmp_path / "shots.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(inner / "b.png", "nested/b.png")
        zf.writestr("__MACOSX/._b.png", b"junk")

    files, warnings = expand_inputs([folder, archive, tmp_path / "missing"], workdir=tmp_path)
    names = sorted(f.path.name for f in files)
    assert names == ["a.png", "nested_b.png"]
    assert any("not found" in w for w in warnings)


def test_zip_path_traversal_is_ignored(tmp_path):
    archive = tmp_path / "evil.zip"
    img = _write(tmp_path / "x.png")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(img, "../../escape.png")
    files, _ = expand_inputs([archive], workdir=tmp_path)
    assert files == []


def test_ingest_populates_session(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    _write(folder / "Screenshot_20240312-100000.png", 10)
    _write(folder / "Screenshot_20240312-100100.png", 20)
    session = Session(workdir=str(tmp_path))
    warnings = ingest(session, [folder])
    assert [r.id for r in session.images] == ["img0000", "img0001"]
    assert session.images[0].width == 10
    assert session.images[0].taken_at == dt.datetime(2024, 3, 12, 10, 0)
    assert warnings == []


def test_ingest_warns_about_duplicate_files(tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    _write(folder / "Screenshot_20240312-100000.png", 10)
    _write(folder / "Screenshot_20240312-100100.png", 10)
    session = Session(workdir=str(tmp_path))
    warnings = ingest(session, [folder])
    assert any("byte-identical" in w for w in warnings)
