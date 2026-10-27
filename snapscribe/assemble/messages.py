"""Build messages out of regions plus their extractions."""

from __future__ import annotations

import datetime as dt
import re

from snapscribe.assemble.timestamps import TimestampResolver, parse_time_token, strip_time
from snapscribe.core.models import (
    AttachmentType,
    Message,
    Region,
    RegionKind,
    Side,
    TimestampPrecision,
)
from snapscribe.core.session import Session
from snapscribe.profiles import Profile

_PLACEHOLDER = re.compile(r"^\s*\[(emoji|image|video|sticker|audio|document)\]\s*$", re.IGNORECASE)
#: What a "delivered"/"read" tick pair looks like once OCR has had a go at it.
_TICK_LINE = re.compile(r"^[\s\\/|<>~^`'\"\u2713\u2714VvWw]{1,5}$")
_DELETED = re.compile(r"\b(this message was deleted|you deleted this message)\b", re.IGNORECASE)


def looks_like_name(candidate: str, max_words: int) -> bool:
    """Could this line be a display name rather than the start of a message?

    Shape only — short, capitalised, no sentence punctuation, at least two
    letters. Necessary but nowhere near sufficient ("Hey" passes), which is why
    :func:`sender_candidates` also requires a name to recur.
    """
    text = candidate.strip().rstrip(":")
    words = text.split()
    if not 1 <= len(words) <= max_words or len(text) > 32:
        return False
    if re.search(r"[.!?,]$", text):
        return False
    if sum(1 for c in text if c.isalpha()) < 2:
        return False
    return all(not word[0].isalpha() or word[0].isupper() for word in words if word)


def sender_candidates(first_lines: list[str], profile: Profile) -> set[str]:
    """Which bubble first lines are really sender names.

    A group chat prints a participant's name above their bubbles, and every
    participant speaks more than once — so a name-shaped line appearing at the
    top of two or more bubbles is a name, while one that appears exactly once
    is the first line of a message. Names listed in the profile are trusted
    outright.
    """
    from collections import Counter

    max_words = profile.senders.sender_line_max_words
    counts = Counter(
        line.strip().rstrip(":") for line in first_lines if looks_like_name(line, max_words)
    )
    known = {k.casefold() for k in profile.senders.known_senders}
    return {name for name, n in counts.items() if n >= 2 or name.casefold() in known}


def split_bubble_text(
    text: str,
    profile: Profile,
    group_chat: bool,
    side: Side,
    allowed_senders: set[str] | None = None,
) -> tuple[str, str, dt.time | None, str]:
    """Split a bubble's raw text into (sender, body, time, raw_time).

    The first line is taken as the sender only when the profile says this app
    prints it there, the line is name-shaped, and — when ``allowed_senders`` is
    given — it is one of the names this conversation actually uses.
    """
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return "", "", None, ""

    sender = ""
    if group_chat and side is Side.LEFT and profile.senders.sender_in_bubble and len(lines) > 1:
        candidate = lines[0].strip().rstrip(":")
        accepted = (
            candidate in allowed_senders
            if allowed_senders is not None
            else looks_like_name(candidate, profile.senders.sender_line_max_words)
        )
        if accepted:
            sender = _snap_sender(candidate, profile.senders.known_senders)
            lines = lines[1:]

    body = "\n".join(lines)
    time, matched = parse_time_token(body, profile.timestamps.time_patterns)
    if time is not None:
        body = strip_time(body, matched)
    body = "\n".join(line for line in body.splitlines() if not _TICK_LINE.match(line)).strip()
    return sender, body, time, matched


def _snap_sender(name: str, known: list[str]) -> str:
    """Snap a recognised name to a known spelling when it is a near match."""
    if not known:
        return name
    from rapidfuzz import process

    match = process.extractOne(name, known, score_cutoff=82)
    return match[0] if match else name


_FILENAME = re.compile(
    r"^\s*\S{1,60}\.(pdf|docx?|xlsx?|pptx?|txt|csv|zip|rar|apk|epub)\s*$", re.IGNORECASE
)
_DURATION = re.compile(r"^\s*\d{1,2}\s*[:.]\s*\d{2}\s*$")


def _attachment_for(region: Region, body: str) -> AttachmentType:
    """Classify a bubble's payload from what the recogniser found in it.

    Deliberately conservative: only signals that are hard to confuse with a
    text message count. A bubble with a photo in it is recognised by the layout
    stage (its rows are too colourful to be text); the rest is textual
    evidence — a bare filename, a bare duration, the deleted-message notice.
    """
    if _DELETED.search(body):
        return AttachmentType.DELETED
    stripped = body.strip()
    if _FILENAME.match(stripped):
        return AttachmentType.DOCUMENT
    if region.meta.get("attachment"):
        return AttachmentType.IMAGE
    if stripped and _DURATION.match(stripped):
        return AttachmentType.AUDIO
    return AttachmentType.NONE


def build_messages(
    session: Session,
    profile: Profile,
    primary_engine: str = "ocr",
    secondary_engine: str = "",
    prefer: str = "primary",
) -> tuple[list[Message], list[str]]:
    """Create one message per bubble region, in capture order.

    Only the clock time printed on each bubble is read here. Dates are resolved
    later, over the *stitched* conversation: consecutive screenshots overlap,
    so within this per-screenshot pass the clock legitimately runs backwards at
    every seam, and a resolver would read that as a new day.
    """
    group_chat = profile.senders.group_chat or bool(session.settings.get("group_chat"))
    contact = "" if group_chat else str(session.settings.get("conversation", ""))
    messages: list[Message] = []
    notes: list[str] = []

    # A group chat prints the sender's name on the first bubble of a run only,
    # so a name carries forward until someone else speaks.
    previous_left_sender = ""
    previous_side: Side | None = None

    # Which names this conversation uses has to be known before any bubble is
    # split, so the first lines are gathered in a cheap pre-pass (dictionary
    # lookups only; nothing is re-recognised).
    allowed = (
        sender_candidates(_first_lines(session, primary_engine), profile) if group_chat else set()
    )

    for ref in session.images:
        regions = [
            r
            for r in session.regions_for(ref.id)
            if r.kind in (RegionKind.BUBBLE, RegionKind.SYSTEM, RegionKind.DATE_SEPARATOR)
        ]
        for region in sorted(regions, key=lambda r: (r.y, r.x)):
            primary = session.extraction(region.id, primary_engine)
            secondary = (
                session.extraction(region.id, secondary_engine) if secondary_engine else None
            )
            ocr_text = primary.text if primary else ""
            vision_text = secondary.text if secondary else ""
            chosen = _choose_text(ocr_text, vision_text, primary, secondary, prefer)

            if region.kind is RegionKind.SYSTEM and _is_date_separator(chosen, profile):
                message = Message(
                    index=len(messages),
                    side=Side.CENTER,
                    body=chosen.strip(),
                    is_system=True,
                    source_image=ref.name,
                    region_id=region.id,
                    timestamp_precision=TimestampPrecision.DATE_ONLY,
                    confidence=primary.confidence if primary else 0.0,
                    ocr_text=ocr_text,
                    vision_text=vision_text,
                )
                message.meta["kind"] = "date_separator"
                messages.append(message)
                continue

            sender, body, time, raw_time = split_bubble_text(
                chosen, profile, group_chat, region.side, allowed_senders=allowed
            )
            attachment = _attachment_for(region, body)
            if attachment is not AttachmentType.NONE and _PLACEHOLDER.match(body):
                body = ""

            if (
                group_chat
                and not sender
                and region.side is Side.LEFT
                and previous_side is Side.LEFT
                and previous_left_sender
            ):
                sender = previous_left_sender
            if not sender and region.kind is not RegionKind.SYSTEM:
                sender = _sender_for_side(region.side, profile, contact)
            if region.side is Side.LEFT:
                previous_left_sender = sender
            if region.kind is not RegionKind.SYSTEM:
                previous_side = region.side

            message = Message(
                index=len(messages),
                conversation=str(session.settings.get("conversation", "")),
                sender="" if region.kind is RegionKind.SYSTEM else sender,
                side=region.side,
                body=body,
                attachment_type=attachment,
                timestamp_raw=raw_time,
                timestamp_precision=TimestampPrecision.NONE,
                is_system=region.kind is RegionKind.SYSTEM,
                confidence=round(primary.confidence, 4) if primary else 0.0,
                agreement=float(primary.meta.get("agreement") or 0.0) if primary else 0.0,
                ocr_text=ocr_text,
                vision_text=vision_text,
                source_image=ref.name,
                region_id=region.id,
            )
            if time is not None:
                message.meta["time"] = time.strftime("%H:%M")
            if primary is not None and primary.error:
                message.flag(f"extraction failed: {primary.error[:80]}")
            messages.append(message)

    return messages, notes


def _first_lines(session: Session, engine: str) -> list[str]:
    """First line of every multi-line left-hand bubble."""
    out: list[str] = []
    for ref in session.images:
        for region in session.regions_for(ref.id):
            if region.kind is not RegionKind.BUBBLE or region.side is not Side.LEFT:
                continue
            extraction = session.extraction(region.id, engine)
            if extraction is None:
                continue
            lines = [line for line in extraction.text.splitlines() if line.strip()]
            if len(lines) > 1:
                out.append(lines[0])
    return out


def _is_date_separator(text: str, profile: Profile) -> bool:
    """A centred pill that reads like a date, not a system notice."""
    cleaned = (text or "").strip()
    if not cleaned or len(cleaned) > 40:
        return False
    for pattern in profile.timestamps.date_separator_patterns:
        try:
            if re.match(pattern, cleaned, re.IGNORECASE):
                return True
        except re.error:
            continue
    from snapscribe.assemble.timestamps import parse_date_separator

    return parse_date_separator(cleaned, dt.date.today(), profile) is not None


def resolve_timestamps(messages: list[Message], profile: Profile, anchor: dt.date) -> list[str]:
    """Attach dates to the stitched conversation, in order.

    Runs once over the whole conversation so a date separator applies to every
    message after it, and so a clock that genuinely runs backwards (a day
    boundary whose separator was clipped away) can be told apart from the
    overlap between two screenshots.
    """
    resolver = TimestampResolver(anchor=anchor, profile=profile)
    for message in messages:
        if message.meta.get("kind") == "date_separator":
            if resolver.apply_separator(message.body.strip()):
                message.timestamp = (
                    dt.datetime.combine(resolver.current_date, dt.time())
                    if resolver.current_date
                    else None
                )
                message.timestamp_precision = TimestampPrecision.DATE_ONLY
            continue
        raw = message.meta.get("time")
        time = None
        if isinstance(raw, str) and raw:
            try:
                hour, minute = raw.split(":")
                time = dt.time(int(hour), int(minute))
            except ValueError:
                time = None
        stamp, precision = resolver.resolve(time)
        message.timestamp = stamp
        message.timestamp_precision = precision
    return resolver.notes


def _choose_text(ocr_text, vision_text, primary, secondary, prefer: str) -> str:
    """Pick which engine's text to use as the body."""
    if prefer == "vision" and vision_text.strip():
        return vision_text
    if prefer == "longest":
        return max((ocr_text, vision_text), key=lambda t: len(t.strip()))
    if ocr_text.strip():
        return ocr_text
    return vision_text


def _sender_for_side(side: Side, profile: Profile, contact: str = "") -> str:
    """Name for a bubble that does not print one.

    In a 1:1 chat the left-hand sender is whoever the header names, which is a
    real name rather than the profile's "Them" placeholder.
    """
    if side is Side.RIGHT:
        return profile.senders.me
    if side is Side.LEFT:
        return contact or profile.senders.other
    return ""


def anchor_date(session: Session) -> dt.date:
    """When the screenshots were taken — the reference for "TODAY"."""
    stamps = [r.taken_at for r in session.images if r.taken_at]
    if stamps:
        return max(stamps).date()
    return dt.date.today()
