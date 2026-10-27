"""Core data model.

``Message`` mirrors the exported CSV columns one-to-one (see
:data:`snapscribe.export.csv_writer.COLUMNS`), so export is a projection rather
than a transformation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Side(StrEnum):
    """Which side of the chat area a bubble sits on."""

    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    UNKNOWN = "unknown"


class RegionKind(StrEnum):
    HEADER = "header"
    FOOTER = "footer"
    CHAT_AREA = "chat_area"
    BUBBLE = "bubble"
    SYSTEM = "system"
    DATE_SEPARATOR = "date_separator"
    SENDER = "sender"
    TIMESTAMP = "timestamp"
    ATTACHMENT = "attachment"
    AVATAR = "avatar"
    DROPPED = "dropped"


class SourceKind(StrEnum):
    SCREENSHOT = "screenshot"
    PHOTO = "photo"
    UNKNOWN = "unknown"


class AttachmentType(StrEnum):
    NONE = ""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    STICKER = "sticker"
    DOCUMENT = "document"
    GIF = "gif"
    LOCATION = "location"
    CONTACT = "contact"
    DELETED = "deleted"
    UNKNOWN = "unknown"


class TimestampPrecision(StrEnum):
    NONE = "none"
    #: Time read off the bubble and a date resolved from a separator/anchor.
    EXACT = "exact"
    #: Time read off the bubble, date assumed from the anchor.
    TIME_ONLY = "time_only"
    #: Date known, time unknown.
    DATE_ONLY = "date_only"
    #: Interpolated between neighbours.
    INFERRED = "inferred"


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Quality:
    """Import-time image quality signals (used by the photo path, phase 8)."""

    blur_score: float = 0.0
    """Variance of the Laplacian; lower means blurrier."""
    glare_ratio: float = 0.0
    """Fraction of pixels in large near-saturated blobs."""
    is_blurry: bool = False
    has_glare: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class ImageRef:
    """One imported image."""

    id: str
    path: str
    index: int = 0
    width: int = 0
    height: int = 0
    taken_at: dt.datetime | None = None
    order_source: str = "unknown"
    """How ``taken_at`` was obtained: exif | filename | mtime | unknown."""
    source_kind: SourceKind = SourceKind.UNKNOWN
    sha1: str = ""
    quality: Quality = field(default_factory=Quality)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        from pathlib import Path

        return Path(self.path).name


@dataclass
class Region:
    """A rectangle of interest on one image, in that image's pixel space."""

    id: str
    image_id: str
    kind: RegionKind = RegionKind.BUBBLE
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    side: Side = Side.UNKNOWN
    order: int = 0
    confidence: float = 1.0
    fill_color: tuple[int, int, int] | None = None
    parent_id: str | None = None
    drop_reason: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def box(self) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)`` with an exclusive bottom-right."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center_x(self) -> float:
        return self.x + self.w / 2

    def crop(self, image: Any, pad: int = 0) -> Any:
        """Crop this region out of a numpy image, clamped to its bounds."""
        h, w = image.shape[:2]
        x0 = max(0, self.x - pad)
        y0 = max(0, self.y - pad)
        x1 = min(w, self.x + self.w + pad)
        y1 = min(h, self.y + self.h + pad)
        return image[y0:y1, x0:x1]


@dataclass
class ChromeInfo:
    """Static UI furniture (status bar, app header, composer) on an image."""

    image_id: str
    header_height: int = 0
    footer_height: int = 0
    left_margin: int = 0
    right_margin: int = 0
    method: str = "unknown"
    """stable_rows | profile | none."""
    confidence: float = 0.0
    header_text: str = ""
    footer_text: str = ""

    def chat_area(self, width: int, height: int) -> Region:
        return Region(
            id=f"{self.image_id}:chat",
            image_id=self.image_id,
            kind=RegionKind.CHAT_AREA,
            x=self.left_margin,
            y=self.header_height,
            w=max(0, width - self.left_margin - self.right_margin),
            h=max(0, height - self.header_height - self.footer_height),
            confidence=self.confidence,
        )


@dataclass
class Extraction:
    """Output of one extractor for one region."""

    region_id: str
    engine: str
    text: str = ""
    confidence: float = 0.0
    duration_ms: float = 0.0
    lines: list[str] = field(default_factory=list)
    line_confidences: list[float] = field(default_factory=list)
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One transcribed chat message. Mirrors the CSV columns."""

    index: int = 0
    conversation: str = ""
    sender: str = ""
    side: Side = Side.UNKNOWN
    body: str = ""
    attachment_type: AttachmentType = AttachmentType.NONE
    timestamp: dt.datetime | None = None
    timestamp_raw: str = ""
    timestamp_precision: TimestampPrecision = TimestampPrecision.NONE
    is_system: bool = False
    confidence: float = 0.0
    agreement: float = 0.0
    needs_review: bool = False
    review_reasons: list[str] = field(default_factory=list)
    ocr_text: str = ""
    vision_text: str = ""
    source_image: str = ""
    region_id: str = ""
    duplicate_of: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def flag(self, reason: str) -> None:
        """Mark for review, de-duplicating reasons."""
        self.needs_review = True
        if reason not in self.review_reasons:
            self.review_reasons.append(reason)
