"""Rule checks over the assembled message list.

These catch the failures that similarity scoring cannot: a sender that
contradicts the bubble's side, timestamps that run backwards, bodies that came
out empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from snapscribe.core.models import AttachmentType, Message, Side


@dataclass
class VerifyConfig:
    """Thresholds. Defaults are a starting point — tune them on your corpus
    with `snapscribe corpus score`, which reports both the share of genuinely
    wrong rows that got flagged and the false-alarm rate."""

    agreement_threshold: float = 0.90
    """Below this similarity between engines, the message needs review."""
    low_confidence: float = 0.75
    """Below this OCR confidence, the message needs review."""
    very_low_confidence: float = 0.55
    min_body_chars: int = 1
    max_body_chars: int = 2000
    allow_backwards_seconds: int = 60
    """Clock skew tolerated between consecutive messages before flagging."""
    flag_unknown_side: bool = True
    flag_empty_text: bool = True


@dataclass
class RuleResult:
    flagged: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def record(self, reason: str) -> None:
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def check_messages(messages: list[Message], config: VerifyConfig | None = None) -> RuleResult:
    """Apply the rule checks in place, returning a summary."""
    cfg = config or VerifyConfig()
    result = RuleResult()

    senders_by_side: dict[Side, set[str]] = {}
    for m in messages:
        if m.sender and m.side in (Side.LEFT, Side.RIGHT):
            senders_by_side.setdefault(m.side, set()).add(m.sender)

    # In a 1:1 chat each side has exactly one sender; more than one means the
    # side classifier and the name parser disagree somewhere.
    right_senders = senders_by_side.get(Side.RIGHT, set())

    previous_ts = None
    for m in messages:
        if cfg.flag_empty_text and not m.body.strip():
            if (
                m.attachment_type in (AttachmentType.NONE, AttachmentType.UNKNOWN)
                and not m.is_system
            ):
                m.flag("empty body")
                result.record("empty body")
        elif len(m.body) > cfg.max_body_chars:
            m.flag("body is implausibly long")
            result.record("body is implausibly long")

        if m.confidence and m.confidence < cfg.very_low_confidence:
            m.flag(f"very low OCR confidence ({m.confidence:.2f})")
            result.record("very low OCR confidence")
        elif m.confidence and m.confidence < cfg.low_confidence:
            m.flag(f"low OCR confidence ({m.confidence:.2f})")
            result.record("low OCR confidence")

        if m.vision_text and m.ocr_text and m.agreement < cfg.agreement_threshold:
            m.flag(f"engines disagree ({m.agreement:.2f})")
            result.record("engines disagree")

        if cfg.flag_unknown_side and m.side is Side.UNKNOWN and not m.is_system:
            m.flag("could not tell which side the bubble is on")
            result.record("unknown side")

        if m.side is Side.LEFT and m.sender and m.sender in right_senders:
            m.flag(f"sender {m.sender!r} appears on both sides")
            result.record("sender on both sides")

        if m.timestamp is not None:
            if previous_ts is not None:
                delta = (m.timestamp - previous_ts).total_seconds()
                if delta < -cfg.allow_backwards_seconds:
                    m.flag("timestamp goes backwards")
                    result.record("timestamp goes backwards")
            previous_ts = m.timestamp

        if m.needs_review:
            result.flagged += 1
    return result
