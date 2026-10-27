"""Timestamps, bubble splitting, stitching, rules and export."""

from __future__ import annotations

import csv
import datetime as dt

import pytest

from snapscribe.assemble.messages import resolve_timestamps, split_bubble_text
from snapscribe.assemble.stitch import find_overlap, stitch_messages
from snapscribe.assemble.timestamps import (
    TimestampResolver,
    parse_date_separator,
    parse_time_token,
    strip_time,
)
from snapscribe.core.models import Message, Side, TimestampPrecision
from snapscribe.export.csv_writer import COLUMNS, write_csv
from snapscribe.profiles import load_profile
from snapscribe.verify.compare import compare_texts, normalize_for_compare
from snapscribe.verify.rules import VerifyConfig, check_messages


@pytest.fixture
def profile():
    return load_profile("whatsapp_light")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello there\n20:14", dt.time(20, 14)),
        ("5:45 PM", dt.time(17, 45)),
        ("12:30 AM", dt.time(0, 30)),
        ("no clock here", None),
        ("99:99", None),
    ],
)
def test_parse_time_token(text, expected):
    assert parse_time_token(text)[0] == expected


def test_parse_time_token_takes_the_last_match():
    """A bubble can mention a time ("meeting at 14:00") and still carry its own
    clock at the end; the clock is the one at the bottom right."""
    time, matched = parse_time_token("Meeting at 14:00\n20:14")
    assert time == dt.time(20, 14)
    assert matched == "20:14"


def test_strip_time_also_removes_read_receipts():
    assert strip_time("Noted 20:15 V/", "20:15") == "Noted"


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("TODAY", dt.date(2024, 3, 12)),
        ("YESTERDAY", dt.date(2024, 3, 11)),
        ("MONDAY, 11 MARCH 2024", dt.date(2024, 3, 11)),
        ("11/03/2024", dt.date(2024, 3, 11)),
        ("not a date at all", None),
    ],
)
def test_parse_date_separator(label, expected, profile):
    assert parse_date_separator(label, dt.date(2024, 3, 12), profile) == expected


def test_resolver_rolls_the_day_when_the_clock_goes_backwards(profile):
    resolver = TimestampResolver(anchor=dt.date(2024, 3, 12), profile=profile)
    resolver.apply_separator("YESTERDAY")
    first, precision = resolver.resolve(dt.time(22, 0))
    assert first == dt.datetime(2024, 3, 11, 22, 0)
    assert precision is TimestampPrecision.EXACT
    second, precision = resolver.resolve(dt.time(8, 0))
    assert second == dt.datetime(2024, 3, 12, 8, 0)
    assert precision is TimestampPrecision.INFERRED
    assert resolver.notes


def test_resolve_timestamps_runs_over_the_stitched_conversation(profile):
    """Resolution happens after stitching: inside a single screenshot pass the
    clock legitimately jumps back at every seam."""
    messages = [
        Message(index=0, body="YESTERDAY", is_system=True, meta={"kind": "date_separator"}),
        Message(index=1, body="morning", meta={"time": "09:00"}),
        Message(index=2, body="evening", meta={"time": "21:00"}),
    ]
    resolve_timestamps(messages, profile, anchor=dt.date(2024, 3, 12))
    assert messages[1].timestamp == dt.datetime(2024, 3, 11, 9, 0)
    assert messages[2].timestamp == dt.datetime(2024, 3, 11, 21, 0)


def test_split_bubble_text_pulls_out_sender_and_clock(profile):
    sender, body, time, raw = split_bubble_text(
        "Priya\nFound a cabin with four bedrooms\n18:20", profile, group_chat=True, side=Side.LEFT
    )
    assert sender == "Priya"
    assert body == "Found a cabin with four bedrooms"
    assert time == dt.time(18, 20)
    assert raw == "18:20"


def test_split_bubble_text_does_not_invent_a_sender_in_a_1to1_chat(profile):
    sender, body, _, _ = split_bubble_text(
        "Sure thing\nsee you then\n18:20", profile, group_chat=False, side=Side.LEFT
    )
    assert sender == ""
    assert body == "Sure thing\nsee you then"


def test_split_bubble_text_keeps_a_sentence_as_body(profile):
    sender, body, _, _ = split_bubble_text(
        "That looks great.\nHow much?\n18:22", profile, group_chat=True, side=Side.LEFT
    )
    assert sender == ""
    assert body.startswith("That looks great.")


def _msg(body, side=Side.LEFT, confidence=0.9):
    return Message(body=body, side=side, confidence=confidence, region_id=f"r-{body[:4]}")


def test_find_overlap_matches_the_repeated_tail():
    tail = [_msg("one"), _msg("two"), _msg("three")]
    head = [_msg("two"), _msg("three"), _msg("four")]
    k, score = find_overlap(tail, head)
    assert k == 2
    assert score > 0.9


def test_stitch_removes_duplicates_and_renumbers():
    merged, report = stitch_messages(
        [
            ("a.png", [_msg("one"), _msg("two")]),
            ("b.png", [_msg("two"), _msg("three")]),
        ]
    )
    assert [m.body for m in merged] == ["one", "two", "three"]
    assert report.removed == 1
    assert [m.index for m in merged] == [0, 1, 2]


def test_stitch_keeps_both_when_there_is_no_overlap():
    merged, report = stitch_messages([("a.png", [_msg("one")]), ("b.png", [_msg("nine")])])
    assert len(merged) == 2
    assert report.notes and "no complete message" in report.notes[0]


def test_stitch_prefers_the_more_confident_reading():
    merged, _ = stitch_messages(
        [
            ("a.png", [_msg("hello there", confidence=0.6)]),
            ("b.png", [_msg("hello there!", confidence=0.95)]),
        ]
    )
    assert merged[0].confidence == pytest.approx(0.95)


def test_normalize_for_compare_folds_ticks_and_placeholders():
    assert normalize_for_compare("Noted [emoji] 20:15 //") == "noted 20:15"


def test_compare_texts_scores_agreement():
    assert compare_texts("hello there", "hello there").agrees
    comparison = compare_texts("hello there", "hello thele")
    assert not comparison.agrees
    assert comparison.only_in_b == ["thele"]


def test_rules_flag_the_real_problems():
    messages = [
        Message(index=0, body="", side=Side.LEFT, confidence=0.9),
        Message(index=1, body="fine", side=Side.UNKNOWN, confidence=0.9),
        Message(index=2, body="low", side=Side.LEFT, confidence=0.4),
        Message(
            index=3,
            body="back in time",
            side=Side.LEFT,
            confidence=0.9,
            timestamp=dt.datetime(2024, 3, 12, 10, 0),
        ),
        Message(
            index=4,
            body="earlier",
            side=Side.LEFT,
            confidence=0.9,
            timestamp=dt.datetime(2024, 3, 12, 9, 0),
        ),
    ]
    result = check_messages(messages, VerifyConfig())
    assert messages[0].needs_review and "empty body" in messages[0].review_reasons
    assert messages[1].needs_review
    assert messages[2].needs_review
    assert messages[4].needs_review
    assert result.flagged == 4


def test_csv_is_written_with_a_bom_so_excel_shows_emoji(tmp_path):
    path = write_csv(
        tmp_path / "out.csv",
        [Message(index=0, body="hei på deg 🎉", sender="Mira", side=Side.LEFT)],
    )
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert rows[0]["body"] == "hei på deg 🎉"
    assert list(rows[0]) == list(COLUMNS)


def test_csv_can_omit_the_debug_columns(tmp_path):
    path = write_csv(tmp_path / "out.csv", [Message(index=0, body="x")], include_debug=False)
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert "ocr_text" not in header


def _group_session(texts, sides, group_chat=True):
    from snapscribe.core.models import Extraction, ImageRef, Region, RegionKind
    from snapscribe.core.session import Session

    session = Session(settings={"group_chat": group_chat})
    session.images.append(ImageRef(id="img0", path="a.png", width=400, height=800))
    session.regions["img0"] = [
        Region(id=f"img0:r{i}", image_id="img0", kind=RegionKind.BUBBLE, y=i * 100, side=side)
        for i, side in enumerate(sides)
    ]
    for i, text in enumerate(texts):
        session.add_extraction(
            Extraction(region_id=f"img0:r{i}", engine="ocr", text=text, confidence=0.99)
        )
    return session


def test_group_chat_sender_carries_across_a_run(profile):
    """WhatsApp prints the name on the first bubble of a run only, so the name
    applies until somebody else speaks."""
    from snapscribe.assemble.messages import build_messages

    session = _group_session(
        [
            "Priya\nFound a cabin\n18:20",
            "I paid the deposit\n18:21",
            "Sure thing\n18:22",
            "Priya\nBooked it\n18:30",
        ],
        [Side.LEFT, Side.LEFT, Side.RIGHT, Side.LEFT],
    )
    messages, _ = build_messages(session, profile)
    assert [m.sender for m in messages] == ["Priya", "Priya", "Me", "Priya"]
    assert messages[1].body.startswith("I paid the deposit")


def test_a_message_opener_is_not_mistaken_for_a_sender(profile):
    """ "Hey" at the top of a bubble is name-shaped. The difference is that a
    real participant's name appears above more than one bubble."""
    from snapscribe.assemble.messages import build_messages

    session = _group_session(
        [
            "Hey\nare you around later?\n18:20",
            "Mira\nyes, after six\n18:21",
            "Mira\nbring the charger\n18:25",
        ],
        [Side.LEFT, Side.LEFT, Side.LEFT],
    )
    messages, _ = build_messages(session, profile)
    assert messages[0].body.startswith("Hey")
    assert messages[1].sender == "Mira"
    assert messages[2].sender == "Mira"


def test_known_senders_are_trusted_on_a_single_appearance(profile):
    """The escape hatch for a participant who speaks only once in the
    screenshots: list them in the profile."""
    from snapscribe.assemble.messages import sender_candidates

    assert sender_candidates(["Tobias"], profile) == set()
    profile.senders.known_senders = ["Tobias"]
    assert sender_candidates(["Tobias"], profile) == {"Tobias"}


def test_a_single_stray_glyph_is_not_treated_as_a_sender(profile):
    sender, body, _, _ = split_bubble_text(
        "O\nSent\n12:40", profile, group_chat=True, side=Side.LEFT
    )
    assert sender == ""
    assert "Sent" in body
