"""Stitching consecutive screenshots together.

Scrolling screenshots overlap: the last few messages of one are the first few
of the next. Aligning on *text* rather than on pixels is what makes this work
for photos of a screen as well as for native screenshots, and it tolerates the
OCR differences between two readings of the same bubble.

The search is over candidate overlap lengths: for each k, score how well the
last k messages of what we have match the first k of the incoming screenshot,
and take the longest k that scores well.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rapidfuzz.fuzz import ratio

from snapscribe.core.models import Message
from snapscribe.verify.compare import normalize_for_compare

DEFAULT_THRESHOLD = 0.82
MAX_OVERLAP = 40


@dataclass
class StitchReport:
    kept: int = 0
    removed: int = 0
    overlaps: list[dict[str, object]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _similar(a: Message, b: Message) -> float:
    ta, tb = normalize_for_compare(a.body), normalize_for_compare(b.body)
    if not ta and not tb:
        # Two media bubbles with no caption: fall back to side and geometry.
        return 0.9 if a.side == b.side else 0.3
    if not ta or not tb:
        return 0.0
    score = ratio(ta, tb) / 100.0
    if a.side != b.side:
        score *= 0.6
    if a.timestamp and b.timestamp and a.timestamp != b.timestamp:
        score *= 0.9
    return score


def find_overlap(
    tail: list[Message], head: list[Message], threshold: float = DEFAULT_THRESHOLD
) -> tuple[int, float]:
    """Longest k where ``tail[-k:]`` matches ``head[:k]``."""
    _best_k, best_score = 0, 0.0
    limit = min(len(tail), len(head), MAX_OVERLAP)
    for k in range(limit, 0, -1):
        scores = [_similar(tail[-k + i], head[i]) for i in range(k)]
        mean = sum(scores) / k
        # A single bad pair inside the window means this is not the overlap.
        if mean >= threshold and min(scores) >= threshold * 0.7:
            return k, mean
        if mean > best_score:
            _best_k, best_score = k, mean
    return 0, best_score


def stitch_messages(
    per_image: list[tuple[str, list[Message]]],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[list[Message], StitchReport]:
    """Concatenate per-screenshot message lists, dropping repeated messages."""
    report = StitchReport()
    merged: list[Message] = []
    for image_name, messages in per_image:
        if not merged:
            merged.extend(messages)
            report.overlaps.append({"image": image_name, "overlap": 0, "score": 1.0})
            continue
        k, score = find_overlap(merged, messages, threshold)
        if k == 0:
            report.notes.append(
                f"{image_name}: shares no complete message with the previous "
                f"screenshot (best candidate scored {score:.2f}). That is normal "
                "when the scroll step is large; it only means nothing could be "
                "de-duplicated across this seam."
            )
        else:
            # Keep whichever reading of each duplicate looks better: higher
            # engine confidence, or the longer body when confidence ties.
            for i in range(k):
                existing = merged[len(merged) - k + i]
                incoming = messages[i]
                if _better(incoming, existing):
                    incoming.duplicate_of = existing.region_id
                    merged[len(merged) - k + i] = incoming
                else:
                    existing.duplicate_of = existing.duplicate_of or incoming.region_id
            report.removed += k
        merged.extend(messages[k:])
        report.overlaps.append({"image": image_name, "overlap": k, "score": round(score, 3)})

    for i, message in enumerate(merged):
        message.index = i
    report.kept = len(merged)
    return merged, report


def _better(candidate: Message, current: Message) -> bool:
    if candidate.confidence > current.confidence + 0.02:
        return True
    if abs(candidate.confidence - current.confidence) <= 0.02:
        return len(candidate.body) > len(current.body)
    return False
