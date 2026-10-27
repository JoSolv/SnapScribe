"""CSV writing.

``utf-8-sig``: without the BOM, Excel on Windows renders emoji and non-English
characters as mojibake, which is exactly the content this tool produces.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from snapscribe.core.models import Message

COLUMNS: tuple[str, ...] = (
    "index",
    "conversation",
    "sender",
    "side",
    "body",
    "attachment_type",
    "timestamp",
    "timestamp_precision",
    "is_system",
    "confidence",
    "agreement",
    "needs_review",
    "review_reasons",
    "source_image",
    "region_id",
    "ocr_text",
    "vision_text",
)

DEBUG_COLUMNS = frozenset({"ocr_text", "vision_text", "region_id"})


def message_to_row(message: Message) -> dict[str, str]:
    return {
        "index": str(message.index),
        "conversation": message.conversation,
        "sender": message.sender,
        "side": message.side.value,
        "body": message.body,
        "attachment_type": message.attachment_type.value,
        "timestamp": message.timestamp.isoformat(sep=" ") if message.timestamp else "",
        "timestamp_precision": message.timestamp_precision.value,
        "is_system": "true" if message.is_system else "false",
        "confidence": f"{message.confidence:.4f}" if message.confidence else "",
        "agreement": f"{message.agreement:.4f}" if message.agreement else "",
        "needs_review": "true" if message.needs_review else "false",
        "review_reasons": "; ".join(message.review_reasons),
        "source_image": message.source_image,
        "region_id": message.region_id,
        "ocr_text": message.ocr_text,
        "vision_text": message.vision_text,
    }


def write_csv(
    path: str | Path,
    messages: Iterable[Message],
    include_debug: bool = True,
    delimiter: str = ",",
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = [c for c in COLUMNS if include_debug or c not in DEBUG_COLUMNS]
    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for message in messages:
            writer.writerow(message_to_row(message))
    return target
