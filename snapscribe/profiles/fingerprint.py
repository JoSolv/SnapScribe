"""Auto-detecting which app and theme a screenshot came from.

Scoring is colour-first so it works without OCR: the background colour, plus
how much of the chat area is covered by each profile's declared bubble
colours. Footer placeholder text is a much stronger signal and is used when the
footer has been read.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from snapscribe.imaging import bgr_to_hex, color_distance, dominant_color, is_dark
from snapscribe.profiles.schema import Profile


@dataclass
class ProfileMatch:
    """Why a profile was (or wasn't) chosen."""

    name: str
    score: float
    signals: dict[str, float] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
            "details": self.details,
        }


def _coverage(chat: np.ndarray, color: tuple[int, int, int], tolerance: int) -> float:
    """Fraction of pixels within ``tolerance`` of ``color``."""
    diff = chat.astype(np.int16) - np.asarray(color, dtype=np.int16)
    dist = np.sqrt((diff.astype(np.float32) ** 2).sum(axis=2))
    return float((dist <= tolerance).mean())


def _ramp(value: float, good: float, bad: float) -> float:
    """1.0 at ``good`` or better, 0.0 at ``bad`` or worse, linear between."""
    if good == bad:
        return 1.0 if value <= good else 0.0
    if good < bad:  # lower is better (distances)
        return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))
    return float(np.clip((value - bad) / (good - bad), 0.0, 1.0))


def score_profile(
    chat_area: np.ndarray,
    profile: Profile,
    footer_text: str = "",
    header_text: str = "",
) -> ProfileMatch:
    """Score one profile against one screenshot's chat area."""
    weights = dict(profile.fingerprint.weights)
    signals: dict[str, float] = {}
    details: dict[str, str] = {}

    bg_measured = dominant_color(chat_area)
    details["measured_background"] = bgr_to_hex(bg_measured)

    bg_expected = profile.colors.bgr("background")
    if bg_expected is not None:
        dist = color_distance(bg_measured, bg_expected)
        signals["background"] = _ramp(dist, good=12.0, bad=90.0)
        details["background_distance"] = f"{dist:.1f}"
    else:
        weights.pop("background", None)

    for key, field_name, floor in (
        ("outgoing", "outgoing_bubble", 0.015),
        ("incoming", "incoming_bubble", 0.015),
    ):
        expected = profile.colors.bgr(field_name)
        if expected is None:
            weights.pop(key, None)
            continue
        cov = _coverage(chat_area, expected, profile.colors.tolerance)
        signals[key] = _ramp(cov, good=0.08, bad=floor)
        details[f"{key}_coverage"] = f"{cov:.3%}"

    if profile.theme in ("light", "dark"):
        measured_dark = is_dark(bg_measured)
        signals["theme"] = 1.0 if measured_dark == (profile.theme == "dark") else 0.0
        details["measured_theme"] = "dark" if measured_dark else "light"
    else:
        weights.pop("theme", None)

    placeholders = profile.fingerprint.footer_placeholders
    if footer_text and placeholders:
        haystack = footer_text.casefold()
        hit = max(
            (1.0 for p in placeholders if p.casefold() in haystack),
            default=0.0,
        )
        signals["footer_text"] = hit
        details["footer_text"] = footer_text[:80]
    else:
        weights.pop("footer_text", None)

    keywords = profile.fingerprint.header_keywords
    if header_text and keywords:
        haystack = header_text.casefold()
        signals["header_text"] = max(
            (1.0 for k in keywords if k.casefold() in haystack), default=0.0
        )
        weights.setdefault("header_text", 0.5)

    total_weight = sum(weights.get(k, 0.0) for k in signals)
    if total_weight <= 0:
        return ProfileMatch(name=profile.name, score=0.0, signals=signals, details=details)
    score = sum(signals[k] * weights.get(k, 0.0) for k in signals) / total_weight
    return ProfileMatch(name=profile.name, score=score, signals=signals, details=details)


def rank_profiles(
    chat_area: np.ndarray,
    profiles: list[Profile],
    footer_text: str = "",
    header_text: str = "",
) -> list[ProfileMatch]:
    """Score every candidate, best first. ``generic`` never outranks a real match."""
    matches = [
        score_profile(chat_area, p, footer_text=footer_text, header_text=header_text)
        for p in profiles
    ]
    return sorted(matches, key=lambda m: (-m.score, m.name == "generic", m.name))


def detect_profile(
    chat_area: np.ndarray,
    profiles: list[Profile],
    footer_text: str = "",
    header_text: str = "",
) -> tuple[Profile, ProfileMatch, list[ProfileMatch]]:
    """Pick the best profile, falling back to ``generic`` below ``min_score``.

    Returns the chosen profile, its match, and the full ranking (which the GUI's
    profile-confirmation dialog in phase 7 will show).
    """
    if not profiles:
        raise ValueError("no profiles to choose from")
    by_name = {p.name: p for p in profiles}
    ranked = rank_profiles(chat_area, profiles, footer_text, header_text)
    candidates = [m for m in ranked if m.name != "generic"] or ranked
    best = candidates[0]
    chosen = by_name[best.name]
    if best.score < chosen.fingerprint.min_score:
        fallback = by_name.get("generic")
        if fallback is not None:
            unsure = ProfileMatch(
                name="generic",
                score=best.score,
                signals=best.signals,
                details={
                    **best.details,
                    "reason": (
                        f"best match {best.name} scored {best.score:.2f}, below its min_score"
                    ),
                },
            )
            return fallback, unsure, ranked
    return chosen, best, ranked
