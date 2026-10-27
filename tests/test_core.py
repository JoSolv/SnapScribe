"""Session serialisation and pipeline ordering."""

from __future__ import annotations

import datetime as dt

import pytest

from snapscribe.core.models import (
    AttachmentType,
    ChromeInfo,
    Extraction,
    ImageRef,
    Message,
    Region,
    RegionKind,
    Side,
    StageStatus,
)
from snapscribe.core.pipeline import PipelineContext, StageError, run_pipeline
from snapscribe.core.session import Session


class _Stage:
    def __init__(self, name: str, requires: tuple[str, ...] = (), fail: bool = False) -> None:
        self.name = name
        self.requires = requires
        self.fail = fail
        self.calls = 0

    def run(self, session: Session, ctx: PipelineContext) -> None:
        self.calls += 1
        if self.fail:
            raise ValueError("boom")
        session.stage(self.name).counts = {"calls": self.calls}


def test_session_round_trip(tmp_path):
    session = Session(name="demo")
    session.images.append(ImageRef(id="img0", path="/tmp/a.png", width=10, height=20))
    session.chrome["img0"] = ChromeInfo(image_id="img0", header_height=5, footer_height=4)
    session.regions["img0"] = [
        Region(id="img0:r0", image_id="img0", kind=RegionKind.BUBBLE, w=3, h=4, side=Side.RIGHT)
    ]
    session.extractions["img0:r0"] = [Extraction(region_id="img0:r0", engine="ocr", text="hi")]
    session.messages.append(
        Message(
            index=0,
            body="hi",
            side=Side.RIGHT,
            attachment_type=AttachmentType.IMAGE,
            timestamp=dt.datetime(2024, 3, 12, 9, 15),
            review_reasons=["low confidence"],
        )
    )
    path = session.save(tmp_path / "s.json")
    loaded = Session.load(path)

    assert loaded.name == "demo"
    assert loaded.images[0].id == "img0"
    assert loaded.chrome["img0"].header_height == 5
    assert loaded.regions["img0"][0].side is Side.RIGHT
    assert loaded.extractions["img0:r0"][0].text == "hi"
    assert loaded.messages[0].timestamp == dt.datetime(2024, 3, 12, 9, 15)
    assert loaded.messages[0].attachment_type is AttachmentType.IMAGE
    assert loaded.messages[0].review_reasons == ["low confidence"]


def test_session_rejects_newer_format(tmp_path):
    path = tmp_path / "s.json"
    path.write_text('{"format_version": 999}', encoding="utf-8")
    with pytest.raises(ValueError, match="newer SnapScribe"):
        Session.load(path)


def test_region_box_and_crop():
    import numpy as np

    region = Region(id="r", image_id="i", x=2, y=3, w=4, h=5)
    assert region.box == (2, 3, 6, 8)
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    assert region.crop(image).shape == (5, 4, 3)
    assert region.crop(image, pad=100).shape == (20, 20, 3)


def test_pipeline_runs_in_order_and_records_status():
    session = Session()
    stages = [_Stage("a"), _Stage("b", requires=("a",))]
    run_pipeline(session, PipelineContext(autosave=False), stages=stages)
    assert session.is_done("a") and session.is_done("b")
    assert session.stages["b"].counts == {"calls": 1}


def test_pipeline_requires_dependencies():
    session = Session()
    stages = [_Stage("a"), _Stage("b", requires=("a",))]
    with pytest.raises(StageError, match="needs a"):
        run_pipeline(session, PipelineContext(autosave=False), stages=stages, from_stage="b")


def test_pipeline_records_failure():
    session = Session()
    stages = [_Stage("a", fail=True)]
    with pytest.raises(ValueError):
        run_pipeline(session, PipelineContext(autosave=False), stages=stages)
    assert session.stages["a"].status is StageStatus.FAILED
    assert "boom" in session.stages["a"].error


def test_invalidate_from_marks_later_stages_pending():
    session = Session()
    for name in ("ingest", "normalize", "layout"):
        session.stage(name).status = StageStatus.DONE
    session.invalidate_from("normalize")
    assert session.is_done("ingest")
    assert not session.is_done("normalize")
    assert not session.is_done("layout")
