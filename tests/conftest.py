"""Shared fixtures: a tiny rendered corpus that every integration test reuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from snapscribe.corpus.render import RenderSpec, build_corpus

TINY_EXPORT = """[11/03/2024, 09:12:04] Jonas: Morning, did you read the draft?
[11/03/2024, 09:14:31] Me: Half of it, the intro is strong
[11/03/2024, 09:15:02] Me: Comments before lunch
[11/03/2024, 09:15:48] Jonas: No rush before 14:00
[12/03/2024, 08:05:12] Jonas: Meeting moved to Thursday
[12/03/2024, 08:30:44] Me: Noted, mornings are fine
"""


@pytest.fixture(scope="session")
def tiny_export(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("export") / "tiny.txt"
    path.write_text(TINY_EXPORT, encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def tiny_corpus(tiny_export: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A two-screenshot light-theme corpus with matching ground truth."""
    out = tmp_path_factory.mktemp("corpus") / "light"
    build_corpus(
        tiny_export,
        out,
        profile_name="whatsapp_light",
        owner="Me",
        spec=RenderSpec(width=720, height=1280, overlap_px=200),
        seed=1,
    )
    return out


@pytest.fixture(scope="session")
def tiny_corpus_dark(tiny_export: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("corpus") / "dark"
    build_corpus(
        tiny_export,
        out,
        profile_name="whatsapp_dark",
        owner="Me",
        spec=RenderSpec(width=720, height=1280, overlap_px=200),
        seed=2,
    )
    return out


def ocr_available() -> bool:
    from snapscribe.extract import get_extractor

    return get_extractor("ocr").available()


needs_ocr = pytest.mark.skipif(not ocr_available(), reason="the [ocr] extra is not installed")
