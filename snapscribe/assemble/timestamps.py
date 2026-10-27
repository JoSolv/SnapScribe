"""Timestamps: read the clock off a bubble, resolve the date around it.

Bubbles carry a time and nothing else. The date comes from the date separators
between them ("TODAY", "YESTERDAY", "Monday, 11 March 2024"), resolved against
an anchor date — normally when the screenshots were taken. ``dateparser``
handles the localised and relative forms so this does not become a table of
month names per language.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from snapscribe.core.models import TimestampPrecision
from snapscribe.profiles import Profile

_FALLBACK_TIME = re.compile(
    r"(?P<h>\d{1,2})\s*[:.]\s*(?P<m>\d{2})\s*(?P<ampm>[APap]\s*\.?\s*[Mm]\.?)?"
)
_RELATIVE = {
    "today": 0,
    "yesterday": 1,
    "i dag": 0,
    "i går": 1,
    "heute": 0,
    "gestern": 1,
    "hoy": 0,
    "ayer": 1,
    "aujourd'hui": 0,
    "hier": 1,
    "oggi": 0,
    "ieri": 1,
    "vandaag": 0,
    "gisteren": 1,
    "idag": 0,
    "igår": 1,
}


def parse_time_token(text: str, patterns: list[str] | None = None) -> tuple[dt.time | None, str]:
    """Find a clock time in ``text``; returns the time and the matched text."""
    candidates = list(patterns or [])
    for pattern in candidates:
        try:
            m = re.search(pattern, text.strip(), re.IGNORECASE)
        except re.error:
            continue
        if m:
            time = _to_time(m.groupdict())
            if time is not None:
                return time, m.group(0)
    best: tuple[dt.time, str] | None = None
    for m in _FALLBACK_TIME.finditer(text):
        time = _to_time(m.groupdict())
        if time is not None:
            best = (time, m.group(0))  # keep the last one: it is the bubble's clock
    return best if best else (None, "")


def _to_time(groups: dict[str, str | None]) -> dt.time | None:
    try:
        hour = int(groups.get("h") or 0)
        minute = int(groups.get("m") or 0)
    except (TypeError, ValueError):
        return None
    ampm = (groups.get("ampm") or "").replace(".", "").replace(" ", "").lower()
    if ampm.startswith("p") and hour < 12:
        hour += 12
    elif ampm.startswith("a") and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return dt.time(hour, minute)


def strip_time(text: str, matched: str) -> str:
    """Remove the clock (and any read-receipt junk after it) from a bubble."""
    if not matched:
        return text.strip()
    index = text.rfind(matched)
    if index < 0:
        return text.strip()
    head = text[:index]
    tail = text[index + len(matched) :]
    if re.fullmatch(r"[\s\\/|<>~^`'\"✓✔Vv]*", tail or ""):
        return head.strip()
    return (head + " " + tail).strip()


def parse_date_separator(
    text: str, anchor: dt.date, profile: Profile | None = None, dayfirst: bool = True
) -> dt.date | None:
    """Resolve a date separator's label to a date."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip(" .,·-")).strip()
    if not cleaned or len(cleaned) > 40:
        return None
    key = cleaned.casefold()
    if key in _RELATIVE:
        return anchor - dt.timedelta(days=_RELATIVE[key])
    languages = None
    if profile is not None and profile.timestamps.locale:
        languages = [profile.timestamps.locale]
    try:
        import dateparser

        parsed = dateparser.parse(
            cleaned,
            languages=languages,
            settings={
                "RELATIVE_BASE": dt.datetime.combine(anchor, dt.time()),
                "DATE_ORDER": "DMY" if dayfirst else "MDY",
                "PREFER_DATES_FROM": "past",
                "REQUIRE_PARTS": ["day", "month"],
            },
        )
    except Exception:
        return None
    if parsed is None:
        return None
    return parsed.date()


@dataclass
class TimestampResolver:
    """Walks the conversation in order, carrying the current date forward."""

    anchor: dt.date
    profile: Profile | None = None
    current_date: dt.date | None = None
    notes: list[str] = field(default_factory=list)
    _last: dt.datetime | None = None

    def apply_separator(self, text: str) -> bool:
        day = parse_date_separator(
            text,
            self.anchor,
            self.profile,
            dayfirst=self.profile.timestamps.dayfirst if self.profile else True,
        )
        if day is None:
            return False
        self.current_date = day
        self._last = None
        return True

    def resolve(self, time: dt.time | None) -> tuple[dt.datetime | None, TimestampPrecision]:
        """Combine a bubble's clock time with the running date."""
        if time is None:
            if self.current_date is None:
                return None, TimestampPrecision.NONE
            return (
                dt.datetime.combine(self.current_date, dt.time()),
                TimestampPrecision.DATE_ONLY,
            )
        if self.current_date is None:
            stamp = dt.datetime.combine(self.anchor, time)
            self._last = stamp
            return stamp, TimestampPrecision.TIME_ONLY
        stamp = dt.datetime.combine(self.current_date, time)
        # A chat runs forwards. A clock that jumps back with no separator in
        # between means the day rolled over and the separator was missed
        # (clipped, or unreadable) — not that the messages are out of order.
        if self._last is not None and stamp < self._last - dt.timedelta(minutes=5):
            self.current_date = self.current_date + dt.timedelta(days=1)
            stamp = dt.datetime.combine(self.current_date, time)
            self.notes.append(
                f"clock went backwards at {time}; assumed the day rolled over to "
                f"{self.current_date.isoformat()}"
            )
            self._last = stamp
            return stamp, TimestampPrecision.INFERRED
        self._last = stamp
        return stamp, TimestampPrecision.EXACT
