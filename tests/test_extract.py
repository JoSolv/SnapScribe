"""Extractor interface, row reconstruction and emoji placeholders."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from snapscribe.core.models import Region
from snapscribe.extract import get_extractor, prepare_crop
from snapscribe.extract.base import ExtractResult, NullExtractor, TextLine
from snapscribe.extract.emoji import emoji_boxes, insert_placeholders
from snapscribe.extract.ocr import _repair_boundary, aggregate_confidence, group_rows, rows_to_text
from snapscribe.extract.vision import clean_response
from tests.conftest import needs_ocr


def test_unknown_engine_is_rejected():
    with pytest.raises(KeyError, match="unknown extractor"):
        get_extractor("telepathy")


def test_null_extractor_keeps_the_pipeline_runnable():
    engine = NullExtractor()
    assert engine.available()
    result = engine.extract(np.zeros((10, 10, 3), np.uint8))
    assert result.text == ""
    assert result.engine == "none"


def test_extract_captures_errors_instead_of_raising():
    class Broken(NullExtractor):
        name = "broken"

        def _extract(self, crop):
            raise RuntimeError("model exploded")

    result = Broken().extract(np.zeros((4, 4, 3), np.uint8))
    assert not result.ok
    assert "model exploded" in result.error


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("Morning! Did y", "you manage to look", "Morning! Did you manage to look"),
        ("Fixeda", "and uploaded", "Fixed and uploaded"),
        ("Thanks. Have a g", "good evening", "Thanks. Have a good evening"),
        ("Perfect, thanks. No", ") rush before 14:00", "Perfect, thanks. No rush before 14:00"),
        ("the end", "end of it", "the end of it"),
    ],
)
def test_repair_boundary_undoes_duplicated_characters(left, right, expected):
    a, b = _repair_boundary(left, right)
    assert f"{a} {b}".strip() == expected


def test_rows_join_same_line_boxes_with_a_space():
    lines = [
        TextLine("Sending that to my", 0.99, (10, 10, 400, 50)),
        TextLine("cardiologist", 0.99, (395, 12, 600, 52)),
        TextLine("20:05", 0.95, (520, 70, 600, 100)),
    ]
    assert rows_to_text(group_rows(lines)) == "Sending that to my cardiologist\n20:05"


def test_rows_keep_non_overlapping_text_untouched():
    lines = [
        TextLine("data", 0.9, (0, 0, 100, 40)),
        TextLine("and more", 0.9, (200, 0, 400, 40)),
    ]
    assert rows_to_text(group_rows(lines)) == "data and more"


def test_aggregate_confidence_weights_by_length():
    lines = [TextLine("ok", 1.0), TextLine("a" * 50, 0.5)]
    assert aggregate_confidence(lines) < 0.6


def test_prepare_crop_pads_blanks_media_and_upscales():
    image = np.full((200, 300, 3), 255, np.uint8)
    image[50:150] = 20  # a "photo" band
    region = Region(id="r", image_id="i", x=0, y=0, w=300, h=200, fill_color=(255, 255, 255))
    region.meta["media_rows"] = [[50, 150]]
    crop, scale = prepare_crop(image, region)
    assert crop.shape[0] >= 200  # padding (and possibly upscaling)
    centre = crop[crop.shape[0] // 2, crop.shape[1] // 2]
    assert tuple(int(c) for c in centre) == (255, 255, 255)
    assert scale >= 1.0


def test_emoji_boxes_finds_colour_that_ocr_did_not_claim():
    crop = np.full((80, 300, 3), 255, np.uint8)
    cv2.circle(crop, (40, 40), 22, (0, 180, 255), -1)  # a saturated blob
    cv2.putText(crop, "hello", (90, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (10, 10, 10), 2)
    boxes = emoji_boxes(crop, text_boxes=[(85, 20, 260, 60)])
    assert len(boxes) == 1
    assert boxes[0][0] < 70


def test_placeholders_land_in_reading_order():
    crop = np.full((80, 300, 3), 255, np.uint8)
    cv2.circle(crop, (30, 40), 20, (0, 180, 255), -1)
    result = ExtractResult(
        text="what was in the sauce?",
        lines=[TextLine("what was in the sauce?", 0.99, (70, 20, 290, 60))],
    )
    out = insert_placeholders(result, crop, placeholder="[emoji]")
    assert out.text == "[emoji] what was in the sauce?"


def test_emoji_policies_other_than_placeholder_leave_text_alone():
    crop = np.full((80, 300, 3), 255, np.uint8)
    cv2.circle(crop, (30, 40), 20, (0, 180, 255), -1)
    result = ExtractResult(text="hi", lines=[TextLine("hi", 0.9, (70, 20, 120, 60))])
    assert insert_placeholders(result, crop, policy="drop").text == "hi"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('Here is the transcription: "hello there"', "hello there"),
        ("The image shows: 20:14", "20:14"),
        ("```\nplain text\n```", "plain text"),
        ("already clean", "already clean"),
    ],
)
def test_vision_response_cleaning(raw, expected):
    assert clean_response(raw) == expected


@needs_ocr
@pytest.mark.ocr
def test_ocr_reads_rendered_text():
    crop = np.full((120, 600, 3), 255, np.uint8)
    cv2.putText(crop, "Hello world", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (10, 10, 10), 3)
    result = get_extractor("ocr").extract(crop)
    assert "hello world" in result.text.casefold()
    assert result.confidence > 0.5
