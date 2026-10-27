"""The corpus itself: export parsing, rendering, and the scorer."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path

from snapscribe.core.models import AttachmentType
from snapscribe.corpus.score import GTMessage, align, load_ground_truth, score_messages
from snapscribe.corpus.whatsapp_export import infer_owner, parse_export

SAMPLE = """[12/03/2024, 17:45:01] Alice: Hey, are we still on for tomorrow?
[12/03/2024, 17:46:12] Me: Yes! 7pm at the usual place
and bring the thing
[12/03/2024, 17:47:00] Alice: <image omitted>
12/03/2024, 17:48 - Bob: Android style line
12/03/2024, 17:49 - Messages and calls are end-to-end encrypted.
[12/03/2024, 5:50:00 PM] Alice: twelve hour clock
"""


def test_parse_export_handles_both_formats():
    messages = parse_export(SAMPLE)
    assert len(messages) == 6
    assert messages[0].timestamp == dt.datetime(2024, 3, 12, 17, 45, 1)
    assert messages[0].sender == "Alice"
    assert messages[1].body.endswith("bring the thing")  # continuation line joined
    assert messages[2].attachment_type is AttachmentType.IMAGE
    assert messages[3].sender == "Bob"
    assert messages[4].is_system and messages[4].sender == ""
    assert messages[5].timestamp == dt.datetime(2024, 3, 12, 17, 50)


def test_a_colon_in_a_sentence_is_not_a_sender():
    messages = parse_export("[12/03/2024, 17:45:01] Alice: Short version: butter goes last\n")
    assert messages[0].sender == "Alice"
    assert messages[0].body == "Short version: butter goes last"


def test_infer_owner_prefers_the_literal_me():
    assert infer_owner(parse_export(SAMPLE)) == "Me"


def test_render_produces_images_and_matching_ground_truth(tiny_corpus: Path):
    images = sorted(tiny_corpus.glob("*.png"))
    assert len(images) >= 2
    meta = json.loads((tiny_corpus / "meta.json").read_text())
    rows = list(csv.DictReader((tiny_corpus / "ground_truth.csv").open(encoding="utf-8-sig")))
    assert len(rows) == meta["message_count"] == 6
    assert rows[0]["sender"] == "Jonas"
    assert rows[0]["side"] == "left"
    assert rows[1]["side"] == "right"
    assert meta["warnings"] == []


def test_every_message_is_fully_visible_somewhere(tiny_corpus: Path):
    """The guarantee that makes recall measurable: no message may be clipped in
    every single screenshot, or no pipeline could ever recover it."""
    meta = json.loads((tiny_corpus / "meta.json").read_text())
    visible = {i for window in meta["visibility"] for i in window["fully_visible"]}
    assert visible == set(range(meta["message_count"]))


def test_screenshots_are_captured_after_the_conversation(tiny_corpus: Path):
    """ "TODAY" is relative to the capture date, so the corpus must look like
    screenshots taken when the conversation ended."""
    meta = json.loads((tiny_corpus / "meta.json").read_text())
    rows = list(csv.DictReader((tiny_corpus / "ground_truth.csv").open(encoding="utf-8-sig")))
    last = max(dt.datetime.fromisoformat(r["timestamp"]) for r in rows if r["timestamp"])
    for name in meta["images"]:
        stamp = name.split("_")[-2]
        assert stamp.startswith(last.strftime("%Y%m%d"))


def test_load_ground_truth_from_an_export(tmp_path: Path):
    path = tmp_path / "export.txt"
    path.write_text(SAMPLE, encoding="utf-8")
    loaded = load_ground_truth(path)
    assert [m.sender for m in loaded][:2] == ["Alice", "Me"]


def test_align_survives_a_missing_message():
    gt = ["one", "two", "three", "four"]
    pred = ["one", "three", "four"]
    pairs = align(gt, pred)
    matched = [(p.gt_index, p.pred_index) for p in pairs if p.pred_index is not None]
    assert matched == [(0, 0), (2, 1), (3, 2)]


def test_score_counts_misses_as_full_errors():
    gt = [GTMessage(0, "Alice", "hello there"), GTMessage(1, "Me", "general kenobi")]
    report = score_messages(gt, [{"sender": "Alice", "body": "hello there", "side": "left"}])
    assert report.matched == 1
    assert report.missed == 1
    assert report.recall == 0.5
    assert report.cer > 0.4  # the missed message counts as entirely wrong


def test_score_matches_attachments_on_type_not_text():
    """An export keeps no text for media, so body similarity is meaningless
    there and the attachment type is compared instead."""
    gt = [GTMessage(0, "Alice", "", attachment_type="image")]
    report = score_messages(gt, [{"sender": "Alice", "body": "", "attachment_type": "image"}])
    assert report.matched == 1
    assert report.attachment_accuracy == 1.0


def test_score_reports_review_queue_quality():
    gt = [GTMessage(0, "A", "right"), GTMessage(1, "A", "wrong")]
    pred = [
        {"sender": "A", "body": "right", "needs_review": "false"},
        {"sender": "A", "body": "wrohg", "needs_review": "true"},
    ]
    report = score_messages(gt, pred)
    assert report.review_catch_rate == 1.0
    assert report.review_false_alarm_rate == 0.0


def test_normalised_cer_ignores_emoji():
    """With the placeholder policy every emoji is a guaranteed character error,
    so the two rates separate text accuracy from the emoji decision."""
    gt = [GTMessage(0, "A", "dinner was unreal 🍝🔥")]
    report = score_messages(gt, [{"sender": "A", "body": "dinner was unreal [emoji] [emoji]"}])
    assert report.cer > 0.3
    assert report.cer_normalized == 0.0
