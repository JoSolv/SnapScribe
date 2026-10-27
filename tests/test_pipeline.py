"""End-to-end runs over the rendered corpus.

The accuracy assertions are deliberate floors, not exact values: they should
catch a regression that breaks a stage without failing on an OCR model update.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from snapscribe.core.models import RegionKind, StageStatus
from snapscribe.core.pipeline import PipelineContext, run_pipeline
from snapscribe.core.session import Session
from snapscribe.corpus.score import score_csv
from tests.conftest import needs_ocr


def _context(corpus: Path, output: Path, engines: list[str], **options) -> PipelineContext:
    return PipelineContext(
        profile_request=options.pop("profile", "auto"),
        debug_dir=options.pop("debug_dir", None),
        options={"inputs": [corpus], "output": output, "engines": engines, **options},
        autosave=False,
    )


def test_layout_only_run_needs_no_ocr(tiny_corpus: Path, tmp_path: Path):
    session = Session()
    ctx = _context(tiny_corpus, tmp_path / "out.csv", ["none"], profile="whatsapp_light")
    run_pipeline(session, ctx, to_stage="layout")

    assert session.is_done("layout")
    assert len(session.images) >= 2
    bubbles = session.all_regions(RegionKind.BUBBLE)
    assert len(bubbles) >= 6
    for image in session.images:
        chrome = session.chrome[image.id]
        assert chrome.header_height > 0
        assert chrome.footer_height > 0
        assert chrome.method == "stable_rows"


def test_overlays_are_written(tiny_corpus: Path, tmp_path: Path):
    debug = tmp_path / "debug"
    session = Session()
    ctx = _context(
        tiny_corpus, tmp_path / "out.csv", ["none"], profile="whatsapp_light", debug_dir=debug
    )
    run_pipeline(session, ctx, to_stage="layout")
    overlays = list(debug.glob("*_layout.png"))
    assert len(overlays) == len(session.images)


def test_stage_can_be_rerun_without_redoing_the_others(tiny_corpus: Path, tmp_path: Path):
    session = Session()
    ctx = _context(tiny_corpus, tmp_path / "out.csv", ["none"], profile="whatsapp_light")
    run_pipeline(session, ctx, to_stage="layout")
    first = session.stages["ingest"].started_at

    ctx.options["profile_overrides"] = {"segmentation.bg_tolerance": 20}
    run_pipeline(session, ctx, from_stage="normalize", to_stage="layout", skip_done=True)
    assert session.stages["ingest"].started_at == first
    assert session.stages["layout"].status is StageStatus.DONE


def test_session_survives_save_and_resume(tiny_corpus: Path, tmp_path: Path):
    path = tmp_path / "session.json"
    session = Session(path=str(path))
    ctx = _context(tiny_corpus, tmp_path / "out.csv", ["none"], profile="whatsapp_light")
    run_pipeline(session, ctx, to_stage="layout")
    session.save()

    resumed = Session.load(path)
    assert resumed.is_done("layout")
    assert len(resumed.all_regions(RegionKind.BUBBLE)) == len(
        session.all_regions(RegionKind.BUBBLE)
    )


@needs_ocr
@pytest.mark.ocr
@pytest.mark.slow
def test_full_run_on_the_light_corpus(tiny_corpus: Path, tmp_path: Path):
    output = tmp_path / "out.csv"
    session = Session()
    run_pipeline(session, _context(tiny_corpus, output, ["ocr"]))

    assert session.profile_name == "whatsapp_light"
    report = score_csv(output, tiny_corpus / "ground_truth.csv")
    assert report.recall == 1.0, report.format_text()
    assert report.precision == 1.0, report.format_text()
    assert report.cer < 0.10, report.format_text()
    assert report.sender_accuracy == 1.0, report.format_text()
    assert report.side_accuracy == 1.0, report.format_text()
    assert report.timestamp_within_minute == 1.0, report.format_text()
    assert report.order_inversions == 0


@needs_ocr
@pytest.mark.ocr
@pytest.mark.slow
def test_full_run_on_the_dark_corpus(tiny_corpus_dark: Path, tmp_path: Path):
    output = tmp_path / "out.csv"
    session = Session()
    run_pipeline(session, _context(tiny_corpus_dark, output, ["ocr"]))

    assert session.profile_name == "whatsapp_dark"
    report = score_csv(output, tiny_corpus_dark / "ground_truth.csv")
    assert report.recall == 1.0, report.format_text()
    assert report.cer < 0.10, report.format_text()
    assert report.side_accuracy == 1.0, report.format_text()


@needs_ocr
@pytest.mark.ocr
@pytest.mark.slow
def test_overlapping_screenshots_do_not_duplicate_messages(tiny_corpus: Path, tmp_path: Path):
    output = tmp_path / "out.csv"
    session = Session()
    run_pipeline(session, _context(tiny_corpus, output, ["ocr"]))
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig")))
    bodies = [r["body"] for r in rows if r["body"].strip()]
    assert len(bodies) == len(set(bodies))


@needs_ocr
@pytest.mark.ocr
@pytest.mark.slow
def test_the_contact_name_comes_from_the_header(tiny_corpus: Path, tmp_path: Path):
    session = Session()
    ctx = _context(tiny_corpus, tmp_path / "out.csv", ["ocr"])
    run_pipeline(session, ctx, to_stage="normalize")
    assert session.settings.get("conversation") == "Jonas"
