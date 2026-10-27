"""Parser for WhatsApp's "Export chat" text format.

This is the ground-truth reader: point it at a real export and you have a
labelled transcript for the screenshots of the same conversation. Both the iOS
bracket form and the Android dash form are accepted, including multi-line
message bodies.

    [12/03/2024, 17:45:01] Alice: Hey                 (iOS)
    12/03/2024, 17:45 - Alice: Hey                    (Android)
    12/03/2024, 17:45 - Messages are end-to-end …     (system: no sender)
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

from snapscribe.core.models import AttachmentType

#: Zero-width characters WhatsApp sprinkles into exports.
_INVISIBLE = dict.fromkeys(map(ord, "‎‏‪‫‬⁦⁧⁨⁩﻿"))

_IOS_RE = re.compile(
    r"^\[(?P<date>[^\],]+),\s*(?P<time>[^\]]+)\]\s*(?P<rest>.*)$",
)
_ANDROID_RE = re.compile(
    r"^(?P<date>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{2,4}),\s*(?P<time>\d{1,2}[:.]\d{2}(?::\d{2})?\s*(?:[APap]\.?[Mm]\.?)?)\s+-\s+(?P<rest>.*)$",
)
_SENDER_RE = re.compile(r"^(?P<sender>[^:]{1,60}):\s?(?P<body>.*)$", re.DOTALL)

_MEDIA_MARKERS: tuple[tuple[str, AttachmentType], ...] = (
    ("image omitted", AttachmentType.IMAGE),
    ("video omitted", AttachmentType.VIDEO),
    ("audio omitted", AttachmentType.AUDIO),
    ("sticker omitted", AttachmentType.STICKER),
    ("gif omitted", AttachmentType.GIF),
    ("document omitted", AttachmentType.DOCUMENT),
    ("contact card omitted", AttachmentType.CONTACT),
    ("media omitted", AttachmentType.UNKNOWN),
    ("this message was deleted", AttachmentType.DELETED),
    ("you deleted this message", AttachmentType.DELETED),
    ("location:", AttachmentType.LOCATION),
)

_SYSTEM_HINTS: tuple[str, ...] = (
    "messages and calls are end-to-end encrypted",
    "created group",
    "added you",
    "changed the subject",
    "changed this group's icon",
    "joined using this group's invite link",
    "left",
    "changed their phone number",
    "missed voice call",
    "missed video call",
    "security code changed",
)

_TIME_FORMATS = (
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",
    "%I:%M:%S%p",
    "%I:%M%p",
)


@dataclass
class ExportMessage:
    """One message from an export — the ground-truth unit."""

    timestamp: dt.datetime | None
    sender: str
    body: str
    is_system: bool = False
    attachment_type: AttachmentType = AttachmentType.NONE
    raw: str = ""
    meta: dict[str, str] = field(default_factory=dict)


def _clean(line: str) -> str:
    return line.translate(_INVISIBLE).replace(" ", " ").rstrip("\n")


def _parse_date(date_text: str, dayfirst: bool = True) -> dt.date | None:
    text = date_text.strip()
    parts = re.split(r"[/.\-]", text)
    if len(parts) != 3:
        return None
    try:
        a, b, c = (int(p) for p in parts)
    except ValueError:
        return None
    if a > 31:  # ISO-ish yyyy/mm/dd
        year, month, day = a, b, c
    else:
        day, month = (a, b) if dayfirst else (b, a)
        if month > 12 and day <= 12:  # the heuristic was wrong, swap back
            day, month = month, day
        year = c
    if year < 100:
        year += 2000
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _parse_time(time_text: str) -> dt.time | None:
    text = time_text.strip().replace(" ", " ").upper().replace(".", ":")
    text = re.sub(r"\s*([AP])M?$", r" \1M", text)
    text = re.sub(r":(?=\s*[AP]M)", "", text)
    for fmt in _TIME_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).time()
        except ValueError:
            continue
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?", text)
    if m:
        try:
            return dt.time(int(m.group(1)) % 24, int(m.group(2)), int(m.group(3) or 0))
        except ValueError:
            return None
    return None


def _classify(body: str) -> tuple[AttachmentType, str]:
    """Detect media placeholders; returns the type and the caption, if any."""
    low = body.casefold()
    for marker, kind in _MEDIA_MARKERS:
        if marker in low:
            caption = re.sub(
                r"[<‎]*\s*(image|video|audio|sticker|gif|document|contact card"
                r"|media)\s*omitted\s*[>]*",
                "",
                body,
                flags=re.IGNORECASE,
            ).strip()
            if kind is AttachmentType.DELETED:
                caption = ""
            return kind, caption
    if re.search(r"\.(jpg|jpeg|png|webp|mp4|opus|m4a|pdf|docx?)\s*\(file attached\)", low):
        ext = re.search(r"\.(\w+)\s*\(file attached\)", low)
        mapping = {
            "jpg": AttachmentType.IMAGE,
            "jpeg": AttachmentType.IMAGE,
            "png": AttachmentType.IMAGE,
            "webp": AttachmentType.STICKER,
            "mp4": AttachmentType.VIDEO,
            "opus": AttachmentType.AUDIO,
            "m4a": AttachmentType.AUDIO,
            "pdf": AttachmentType.DOCUMENT,
            "doc": AttachmentType.DOCUMENT,
            "docx": AttachmentType.DOCUMENT,
        }
        kind = mapping.get(ext.group(1) if ext else "", AttachmentType.UNKNOWN)
        caption = re.sub(r"^\S+\s*\(file attached\)\s*", "", body).strip()
        return kind, caption
    return AttachmentType.NONE, body


def _looks_system(body: str) -> bool:
    low = body.casefold()
    return any(h in low for h in _SYSTEM_HINTS)


def parse_export(text: str, dayfirst: bool = True) -> list[ExportMessage]:
    """Parse export text into messages, joining continuation lines."""
    messages: list[ExportMessage] = []
    pending: list[str] = []
    header: tuple[dt.date | None, dt.time | None, str] | None = None

    def flush() -> None:
        nonlocal pending, header
        if header is None:
            pending = []
            return
        date, time, rest = header
        body_text = "\n".join(pending).strip() if pending else ""
        full = f"{rest}\n{body_text}".strip() if body_text else rest
        sender = ""
        body = full
        m = _SENDER_RE.match(full)
        if m and not _looks_system(full):
            candidate = m.group("sender").strip()
            # A colon inside a sentence is not a sender; real senders are short
            # and have no sentence punctuation.
            if len(candidate.split()) <= 6 and not re.search(r"[.!?]$", candidate):
                sender = candidate
                body = m.group("body").strip()
        is_system = not sender or _looks_system(full)
        attachment, caption = _classify(body)
        stamp = dt.datetime.combine(date, time) if date and time else None
        messages.append(
            ExportMessage(
                timestamp=stamp,
                sender="" if is_system else sender,
                body=caption if attachment is not AttachmentType.NONE else body,
                is_system=is_system,
                attachment_type=attachment,
                raw=full,
            )
        )
        pending = []
        header = None

    for raw_line in text.splitlines():
        line = _clean(raw_line)
        if not line.strip() and header is None:
            continue
        m = _IOS_RE.match(line) or _ANDROID_RE.match(line)
        if m:
            flush()
            header = (
                _parse_date(m.group("date"), dayfirst),
                _parse_time(m.group("time")),
                m.group("rest").strip(),
            )
        elif header is not None:
            pending.append(line)
    flush()
    return messages


def parse_export_file(path: str | Path, dayfirst: bool = True) -> list[ExportMessage]:
    data = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    return parse_export(data, dayfirst=dayfirst)


def infer_owner(messages: list[ExportMessage]) -> str:
    """Guess which sender is the phone's owner.

    An export does not say who "me" is. In practice the owner is the most
    frequent sender in a 1:1 chat only half the time, so this is a hint the
    caller can override with ``--me``.
    """
    counts: dict[str, int] = {}
    for m in messages:
        if m.sender:
            counts[m.sender] = counts.get(m.sender, 0) + 1
    if not counts:
        return ""
    for literal in ("Me", "me", "You", "you"):
        if literal in counts:
            return literal
    return max(counts.items(), key=lambda kv: kv[1])[0]
