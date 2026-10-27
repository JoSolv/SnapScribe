"""Scoring a transcript CSV against ground truth.

The metrics are the ones that matter for this problem:

* **CER / WER** over aligned message bodies — how good is the text?
* **recall / precision** — were messages found at all, or invented?
* **sender accuracy** — did left/right and name attribution work?
* **timestamp accuracy** — exact, and within a minute.

Predicted and ground-truth messages are aligned with a Needleman–Wunsch pass
over body similarity, so one missed message does not shift every later pair and
tank the score.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio

GAP_PENALTY = -0.35
MATCH_FLOOR = 0.30
"""Pairs below this similarity are left unaligned rather than forced together."""


@dataclass
class GTMessage:
    """Ground truth, loaded from an export or a generated corpus CSV."""

    index: int
    sender: str
    body: str
    timestamp: dt.datetime | None = None
    attachment_type: str = ""
    is_system: bool = False


@dataclass
class Pair:
    gt_index: int | None
    pred_index: int | None
    similarity: float = 0.0


@dataclass
class ScoreReport:
    corpus: str = ""
    predicted: str = ""
    gt_count: int = 0
    pred_count: int = 0
    matched: int = 0
    missed: int = 0
    spurious: int = 0
    recall: float = 0.0
    precision: float = 0.0
    f1: float = 0.0
    cer: float = 1.0
    cer_normalized: float = 1.0
    wer: float = 1.0
    exact_body: float = 0.0
    sender_accuracy: float = 0.0
    side_accuracy: float = 0.0
    timestamp_exact: float = 0.0
    timestamp_within_minute: float = 0.0
    attachment_accuracy: float = 0.0
    order_inversions: int = 0
    needs_review_count: int = 0
    review_catch_rate: float = 0.0
    """Share of genuinely wrong messages that were flagged for review."""
    review_false_alarm_rate: float = 0.0
    worst: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def format_text(self) -> str:
        pct = lambda v: f"{v * 100:5.1f}%"  # noqa: E731
        lines = [
            f"corpus:     {self.corpus}",
            f"predicted:  {self.predicted}",
            "",
            f"messages    ground truth {self.gt_count}, predicted {self.pred_count}",
            f"            matched {self.matched}, missed {self.missed}, spurious {self.spurious}",
            f"recall      {pct(self.recall)}   precision {pct(self.precision)}   f1 {pct(self.f1)}",
            "",
            f"CER         {pct(self.cer)}  "
            f"({pct(self.cer_normalized)} ignoring emoji and punctuation)",
            f"WER         {pct(self.wer)}",
            f"exact body  {pct(self.exact_body)}",
            "",
            f"sender      {pct(self.sender_accuracy)}",
            f"side        {pct(self.side_accuracy)}",
            f"timestamp   within 1 min {pct(self.timestamp_within_minute)}  "
            f"(to the second {pct(self.timestamp_exact)}; a screenshot rarely shows seconds)",
            f"attachment  {pct(self.attachment_accuracy)}",
            f"order       {self.order_inversions} inversion(s)",
            "",
            f"review queue {self.needs_review_count} flagged; "
            f"catches {pct(self.review_catch_rate)} of wrong rows, "
            f"false alarms {pct(self.review_false_alarm_rate)}",
        ]
        if self.worst:
            lines.append("")
            lines.append("worst rows:")
            for row in self.worst:
                lines.append(f"  [{row['kind']}] gt={row['gt']!r}")
                lines.append(f"              pred={row['pred']!r}")
        return "\n".join(lines)


# ---------------------------------------------------------------- normalising

_WS_RE = re.compile(r"\s+")
_PUNCT_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "…": "...",
        " ": " ",
    }
)


_EMOJI_RE = re.compile(
    "[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\u2190-\u21ff\ufe00-\ufe0f\u200d]+"
)
_PLACEHOLDER_RE = re.compile(r"\[(emoji|image|video|sticker|audio|document)\]", re.IGNORECASE)


def normalize_text(text: str, drop_punctuation: bool = False) -> str:
    """Fold away differences nobody should be penalised for."""
    out = unicodedata.normalize("NFKC", text or "").translate(_PUNCT_MAP)
    out = _WS_RE.sub(" ", out).strip().casefold()
    if drop_punctuation:
        out = re.sub(r"[^\w\s]", "", out)
        out = _WS_RE.sub(" ", out).strip()
    return out


def strip_emoji(text: str) -> str:
    """Remove emoji and the placeholders that stand in for them.

    Classical OCR cannot read emoji, so with the default ``placeholder`` policy
    every emoji is a guaranteed character error. Reporting one error rate that
    includes them and one that does not separates "how well does it read text"
    from "what did we decide to do about emoji".
    """
    out = _PLACEHOLDER_RE.sub(" ", text or "")
    out = _EMOJI_RE.sub(" ", out)
    return _WS_RE.sub(" ", out).strip()


def _parse_dt(value: str) -> dt.datetime | None:
    if not value:
        return None
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return (
                dt.datetime.fromisoformat(value)
                if fmt is None
                else dt.datetime.strptime(value, fmt)
            )
        except ValueError:
            continue
    return None


def _truthy(value: str) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def load_ground_truth(path: str | Path) -> list[GTMessage]:
    """Load ground truth from a CSV (generated corpus) or a WhatsApp export."""
    p = Path(path)
    if p.suffix.lower() in (".txt", ".log"):
        from snapscribe.corpus.whatsapp_export import parse_export_file

        return [
            GTMessage(
                index=i,
                sender=m.sender,
                body=m.body,
                timestamp=m.timestamp,
                attachment_type=m.attachment_type.value,
                is_system=m.is_system,
            )
            for i, m in enumerate(parse_export_file(p))
        ]
    rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
    return [
        GTMessage(
            index=i,
            sender=(r.get("sender") or "").strip(),
            body=r.get("body") or "",
            timestamp=_parse_dt((r.get("timestamp") or "").strip()),
            attachment_type=(r.get("attachment_type") or "").strip(),
            is_system=_truthy(r.get("is_system", "")),
        )
        for i, r in enumerate(rows)
    ]


def load_predicted(path: str | Path) -> list[dict[str, str]]:
    return list(csv.DictReader(Path(path).open(encoding="utf-8-sig")))


# ----------------------------------------------------------------- alignment


def _similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return ratio(na, nb) / 100.0


def _pair_similarity(
    gt_body: str, gt_attachment: str, pred_body: str, pred_attachment: str
) -> float:
    """Similarity of one candidate pair.

    A WhatsApp export writes "<image omitted>" and keeps no text, so a media
    message has an empty ground-truth body. Scoring that against whatever the
    pipeline read off the bubble would be meaningless, so attachment-only rows
    are matched on their *type* instead.
    """
    gt_text = strip_emoji(normalize_text(gt_body))
    pred_text = strip_emoji(normalize_text(pred_body))
    if not gt_text and not pred_text and (normalize_text(gt_body) or normalize_text(pred_body)):
        # Both sides are emoji (or placeholders standing in for them) and
        # nothing else: there is no text to compare, so treat them as the same
        # message and let the CER columns account for the difference.
        return 0.88

    gt_empty = not normalize_text(gt_body)
    if gt_empty and gt_attachment:
        if pred_attachment and pred_attachment.casefold() != "none":
            return 0.9
        return 0.5 if not normalize_text(pred_body) else 0.35
    return _similarity(gt_body, pred_body)


def align(
    gt_bodies: list[str],
    pred_bodies: list[str],
    gt_attachments: list[str] | None = None,
    pred_attachments: list[str] | None = None,
) -> list[Pair]:
    """Global alignment maximising total body similarity."""
    n, m = len(gt_bodies), len(pred_bodies)
    ga = gt_attachments or [""] * n
    pa = pred_attachments or [""] * m
    sim = [
        [_pair_similarity(gt_bodies[i], ga[i], pred_bodies[j], pa[j]) for j in range(m)]
        for i in range(n)
    ]
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP_PENALTY
        back[i][0] = "up"
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP_PENALTY
        back[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = sim[i - 1][j - 1]
            diag = score[i - 1][j - 1] + (s if s >= MATCH_FLOOR else 2 * GAP_PENALTY)
            up = score[i - 1][j] + GAP_PENALTY
            left = score[i][j - 1] + GAP_PENALTY
            best = max(diag, up, left)
            score[i][j] = best
            back[i][j] = "diag" if best == diag else ("up" if best == up else "left")

    pairs: list[Pair] = []
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j]
        if move == "diag" and i > 0 and j > 0:
            s = sim[i - 1][j - 1]
            if s >= MATCH_FLOOR:
                pairs.append(Pair(i - 1, j - 1, s))
            else:
                pairs.append(Pair(i - 1, None))
                pairs.append(Pair(None, j - 1))
            i, j = i - 1, j - 1
        elif move == "up" and i > 0:
            pairs.append(Pair(i - 1, None))
            i -= 1
        elif j > 0:
            pairs.append(Pair(None, j - 1))
            j -= 1
        else:  # pragma: no cover - defensive
            break
    pairs.reverse()
    return pairs


# ------------------------------------------------------------------- scoring


def score_messages(
    gt: list[GTMessage],
    pred: list[dict[str, str]],
    include_system: bool = False,
    corpus: str = "",
    predicted: str = "",
    worst_n: int = 5,
) -> ScoreReport:
    """Compare predicted rows against ground truth."""
    if not include_system:
        gt = [g for g in gt if not g.is_system]
        pred = [p for p in pred if not _truthy(p.get("is_system", ""))]

    report = ScoreReport(corpus=corpus, predicted=predicted, gt_count=len(gt), pred_count=len(pred))
    if not gt:
        return report

    pairs = align(
        [g.body for g in gt],
        [p.get("body", "") for p in pred],
        [g.attachment_type for g in gt],
        [(p.get("attachment_type") or "") for p in pred],
    )
    matched = [p for p in pairs if p.gt_index is not None and p.pred_index is not None]
    report.matched = len(matched)
    report.missed = sum(1 for p in pairs if p.gt_index is not None and p.pred_index is None)
    report.spurious = sum(1 for p in pairs if p.gt_index is None and p.pred_index is not None)
    report.recall = report.matched / len(gt)
    report.precision = report.matched / len(pred) if pred else 0.0
    if report.recall + report.precision:
        report.f1 = 2 * report.recall * report.precision / (report.recall + report.precision)

    # Text error rates are computed over ground-truth characters, counting a
    # missed message as fully wrong — otherwise dropping hard messages would
    # "improve" the CER.
    gt_chars = sum(len(g.body) for g in gt) or 1
    gt_chars_norm = sum(len(strip_emoji(normalize_text(g.body))) for g in gt) or 1
    gt_words = sum(len(normalize_text(g.body).split()) for g in gt) or 1
    char_errors = 0
    char_errors_norm = 0
    word_errors = 0
    exact = 0
    sender_ok = 0
    side_ok = 0
    side_total = 0
    ts_exact = 0
    ts_minute = 0
    ts_total = 0
    att_ok = 0
    wrong_rows: list[int] = []
    flagged_rows: list[int] = []
    rows: list[dict[str, Any]] = []

    for pair in matched:
        g = gt[pair.gt_index]  # type: ignore[index]
        p = pred[pair.pred_index]  # type: ignore[index]
        pbody = p.get("body", "")
        if g.attachment_type and not normalize_text(g.body):
            # Nothing to compare: the export kept no text for this message.
            att_ok += int(bool((p.get("attachment_type") or "").strip()))
            exact += 1
            sender_ok += int(normalize_text(p.get("sender") or "") == normalize_text(g.sender))
            if g.timestamp is not None:
                ts_total += 1
                pts = _parse_dt((p.get("timestamp") or "").strip())
                if pts is not None:
                    delta = abs((pts - g.timestamp).total_seconds())
                    ts_exact += int(delta < 1)
                    ts_minute += int(delta <= 60)
            if _truthy(p.get("needs_review", "")):
                flagged_rows.append(pair.pred_index)  # type: ignore[arg-type]
            continue
        char_errors += Levenshtein.distance(g.body, pbody)
        char_errors_norm += Levenshtein.distance(
            strip_emoji(normalize_text(g.body)), strip_emoji(normalize_text(pbody))
        )
        word_errors += Levenshtein.distance(
            normalize_text(g.body).split(), normalize_text(pbody).split()
        )
        row_exact = normalize_text(g.body) == normalize_text(pbody)
        exact += int(row_exact)

        psender = (p.get("sender") or "").strip()
        sender_match = normalize_text(psender) == normalize_text(g.sender)
        sender_ok += int(sender_match)

        pside = (p.get("side") or "").strip().casefold()
        if pside in ("left", "right"):
            side_total += 1
            # Ground truth records the owner as the "me" sender; a right-side
            # bubble should be the owner.
            expected = "right" if normalize_text(g.sender) == normalize_text(_owner(gt)) else "left"
            side_ok += int(pside == expected)

        if g.timestamp is not None:
            ts_total += 1
            pts = _parse_dt((p.get("timestamp") or "").strip())
            if pts is not None:
                delta = abs((pts - g.timestamp).total_seconds())
                ts_exact += int(delta < 1)
                ts_minute += int(delta <= 60)

        att_ok += int(
            (p.get("attachment_type") or "").strip().casefold()
            == (g.attachment_type or "").strip().casefold()
        )

        if _truthy(p.get("needs_review", "")):
            flagged_rows.append(pair.pred_index)  # type: ignore[arg-type]
        if not row_exact:
            wrong_rows.append(pair.pred_index)  # type: ignore[arg-type]
            rows.append(
                {
                    "kind": "text",
                    "gt": g.body[:120],
                    "pred": pbody[:120],
                    "similarity": round(pair.similarity, 3),
                }
            )

    for pair in pairs:
        if pair.gt_index is not None and pair.pred_index is None:
            g = gt[pair.gt_index]
            char_errors += len(g.body)
            char_errors_norm += len(strip_emoji(normalize_text(g.body)))
            word_errors += len(normalize_text(g.body).split())
            rows.append({"kind": "missed", "gt": g.body[:120], "pred": "", "similarity": 0.0})
        elif pair.pred_index is not None and pair.gt_index is None:
            rows.append(
                {
                    "kind": "spurious",
                    "gt": "",
                    "pred": pred[pair.pred_index].get("body", "")[:120],
                    "similarity": 0.0,
                }
            )

    report.cer = char_errors / gt_chars
    report.cer_normalized = char_errors_norm / gt_chars_norm
    report.wer = word_errors / gt_words
    denom = report.matched or 1
    report.exact_body = exact / denom
    report.sender_accuracy = sender_ok / denom
    report.side_accuracy = side_ok / side_total if side_total else 0.0
    report.timestamp_exact = ts_exact / ts_total if ts_total else 0.0
    report.timestamp_within_minute = ts_minute / ts_total if ts_total else 0.0
    report.attachment_accuracy = att_ok / denom
    report.order_inversions = _count_inversions([p.pred_index for p in matched])

    report.needs_review_count = sum(1 for p in pred if _truthy(p.get("needs_review", "")))
    wrong = set(wrong_rows)
    flagged = set(flagged_rows)
    report.review_catch_rate = len(wrong & flagged) / len(wrong) if wrong else 1.0
    clean = denom - len(wrong)
    report.review_false_alarm_rate = (len(flagged - wrong) / clean) if clean > 0 else 0.0
    report.worst = sorted(rows, key=lambda r: r["similarity"])[:worst_n]
    return report


def _owner(gt: list[GTMessage]) -> str:
    """Whoever the ground truth calls the device owner (``Me`` by convention)."""
    names = {g.sender for g in gt if g.sender}
    for candidate in ("Me", "me", "You", "you"):
        if candidate in names:
            return candidate
    return ""


def _count_inversions(seq: list[int | None]) -> int:
    values = [v for v in seq if v is not None]
    return sum(1 for i in range(len(values) - 1) if values[i + 1] < values[i])


def score_csv(
    predicted_csv: str | Path,
    ground_truth: str | Path,
    include_system: bool = False,
    worst_n: int = 5,
) -> ScoreReport:
    gt = load_ground_truth(ground_truth)
    pred = load_predicted(predicted_csv)
    return score_messages(
        gt,
        pred,
        include_system=include_system,
        corpus=str(ground_truth),
        predicted=str(predicted_csv),
        worst_n=worst_n,
    )


def write_report(report: ScoreReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return p
