"""Comparing two transcriptions of the same bubble."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

_WS = re.compile(r"\s+")
_PUNCT = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
        " ": " ",
        " ": " ",
    }
)
#: Read receipts ("delivered"/"read" ticks) land at a bubble's tail as
#: punctuation-ish junk: "V/", "//", "~". A lone trailing "V" is left
#: alone (it could be real text), but "V" plus a slash is always a tick.
_TICK_JUNK = re.compile(r"(?:\s*[Vv]?[\\/|<>~^`'\"\u2713\u2714]+)+\s*$")


def normalize_for_compare(text: str, drop_emoji_placeholders: bool = True) -> str:
    """Fold away everything the two engines should not be judged on."""
    out = unicodedata.normalize("NFKC", text or "").translate(_PUNCT)
    if drop_emoji_placeholders:
        out = re.sub(r"\[(emoji|image|video|sticker|audio)\]", " ", out, flags=re.IGNORECASE)
    out = _TICK_JUNK.sub("", out)
    out = _WS.sub(" ", out).strip().casefold()
    return out


@dataclass
class Comparison:
    similarity: float = 0.0
    """0..1 over normalised text."""
    token_similarity: float = 0.0
    distance: int = 0
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    a_empty: bool = False
    b_empty: bool = False

    @property
    def agrees(self) -> bool:
        return self.similarity >= 0.95


def word_diff(a: str, b: str) -> tuple[list[str], list[str]]:
    """Words present in only one of the two texts (order-insensitive)."""
    from collections import Counter

    ca = Counter(normalize_for_compare(a).split())
    cb = Counter(normalize_for_compare(b).split())
    only_a = sorted((ca - cb).elements())
    only_b = sorted((cb - ca).elements())
    return only_a, only_b


def compare_texts(a: str, b: str) -> Comparison:
    na, nb = normalize_for_compare(a), normalize_for_compare(b)
    if not na and not nb:
        return Comparison(similarity=1.0, token_similarity=1.0, a_empty=True, b_empty=True)
    if not na or not nb:
        return Comparison(
            similarity=0.0,
            token_similarity=0.0,
            distance=max(len(na), len(nb)),
            only_in_a=na.split(),
            only_in_b=nb.split(),
            a_empty=not na,
            b_empty=not nb,
        )
    only_a, only_b = word_diff(a, b)
    return Comparison(
        similarity=fuzz.ratio(na, nb) / 100.0,
        token_similarity=fuzz.token_sort_ratio(na, nb) / 100.0,
        distance=Levenshtein.distance(na, nb),
        only_in_a=only_a,
        only_in_b=only_b,
    )
