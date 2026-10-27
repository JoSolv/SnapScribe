"""Reading the conversation title out of the app header.

The header names who you are talking to. In a 1:1 chat that is the sender of
every left-hand bubble, which is the difference between a transcript that says
"Them" and one that says "Jonas".
"""

from __future__ import annotations

import re

_STATUS_WORDS = (
    "online",
    "offline",
    "typing",
    "last seen",
    "tap here for",
    "click here for",
    "contact info",
    "group info",
    "participants",
    "members",
    "you,",
    "active now",
)
_TIME_LIKE = re.compile(r"^\s*\d{1,2}\s*[:.]\s*\d{2}\s*(?:[APap]\.?[Mm]\.?)?\s*$")
_NUMERIC = re.compile(r"^[\s\d\W]+$")
_LEADING_CLOCK = re.compile(r"^\s*\d{1,2}\s*[:.]\s*\d{2}\s*(?:[APap]\.?[Mm]\.?)?\b")


STATUS_FRACTION = 0.34
"""Share of the header occupied by the device status bar."""


def app_bar_text(lines: list[tuple[str, int]], header_height: int) -> str:
    """Keep only the header lines below the status bar.

    Cropping the status bar away before recognition would be simpler, but text
    detectors resize their input and a narrower crop changes what they find —
    on a 1080px header, cropping made the contact name disappear entirely. So
    the whole header is recognised and the status bar is filtered out by the
    position of each line instead.
    """
    cutoff = header_height * STATUS_FRACTION
    return "\n".join(text for text, top in lines if top >= cutoff)


def parse_header_title(text: str, max_words: int = 6) -> str:
    """Pick the conversation title out of OCR'd header text.

    Drops the status-bar clock, presence lines ("online", "last seen …") and
    battery/signal noise, then takes the longest remaining line.
    """
    best = ""
    for raw in (text or "").splitlines():
        line = _LEADING_CLOCK.sub("", raw).strip(" \t·.-")
        if not line or len(line) > 48:
            continue
        low = line.casefold()
        if _TIME_LIKE.match(line) or _NUMERIC.match(line):
            continue
        if any(low.startswith(word) or low == word.strip() for word in _STATUS_WORDS):
            continue
        if len(line.split()) > max_words:
            continue
        # Mostly digits and punctuation means battery/signal noise, not a name.
        word_chars = sum(1 for c in line if c.isalpha())
        if word_chars < max(2, len(line) * 0.5):
            continue
        if len(line) > len(best):
            best = line
    return best.strip()
