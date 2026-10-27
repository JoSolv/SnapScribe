"""Profile schema.

Profiles are user-authored YAML, so they are the one place where pydantic
earns its keep: a typo in a colour or a negative fraction should produce a
pointed error, not a mystifying failure three stages later.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from snapscribe.imaging import hex_to_bgr

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
HexColor = Annotated[str, Field(pattern=r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")]
Theme = Literal["light", "dark", "auto"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class LayoutConfig(_Base):
    """Geometry. Fractions are of the full image unless noted."""

    header_fraction: Fraction = 0.10
    """Fallback header height when stable-row detection can't run."""
    footer_fraction: Fraction = 0.10
    left_margin_fraction: Fraction = 0.0
    right_margin_fraction: Fraction = 0.0
    side_split: Fraction = 0.5
    """Chat-area-relative x that divides incoming from outgoing."""
    side_deadzone: Fraction = 0.07
    """Band around ``side_split`` where colour decides instead of position."""
    edge_tolerance_px: int = Field(default=3, ge=0)
    """Bubbles touching this close to the chat-area top/bottom are clipped."""
    rounded_bubbles: bool = True
    """Bubbles have rounded corners, so a flat cut edge means the bubble was
    truncated by the viewport rather than ending there."""
    clip_zone_fraction: Fraction = 0.06
    """How far from the chat-area edge the flat-edge test still applies."""
    avatar_width_fraction: Fraction = 0.0
    """Width of the avatar gutter in group chats, if the app shows one."""


class ColorConfig(_Base):
    background: HexColor | None = None
    incoming_bubble: HexColor | None = None
    outgoing_bubble: HexColor | None = None
    system_bubble: HexColor | None = None
    text: HexColor | None = None
    tolerance: int = Field(default=42, ge=1, le=255)
    """Max BGR distance still counted as "this colour"."""

    def bgr(self, field: str) -> tuple[int, int, int] | None:
        value = getattr(self, field, None)
        return hex_to_bgr(value) if value else None


class SegmentationConfig(_Base):
    """Knobs for the bubble segmenter (see :mod:`snapscribe.layout.bubbles`)."""

    bg_tolerance: int = Field(default=18, ge=1, le=255)
    """How far a pixel must be from the background to count as content."""
    close_kernel: tuple[int, int] = (31, 9)
    """Horizontal-leaning close kernel; merges words and lines into one blob."""
    open_kernel: tuple[int, int] = (3, 3)
    dilate_px: int = Field(default=0, ge=0)
    min_width_fraction: Fraction = 0.06
    max_width_fraction: Fraction = 0.98
    min_height_px: int = Field(default=16, ge=1)
    max_height_fraction: Fraction = 0.75
    min_area_px: int = Field(default=400, ge=1)
    row_gap_px: int = Field(default=6, ge=0)
    """Blank rows needed to split two vertically adjacent bubbles."""
    attachment_variance: float = Field(default=1400.0, ge=0.0)
    """Colour variance above which a bubble row is treated as media, not text."""
    attachment_color_count: int = Field(default=9, ge=2)
    """Distinct quantised colours above which a row is media rather than text.

    A text row holds the bubble fill, the text colour and the antialiasing
    between them — a handful of colours. A photo row holds dozens."""
    attachment_min_rows: int = Field(default=40, ge=1)
    """Shortest run of media-looking rows that counts as an attachment."""
    attachment_min_area_fraction: Fraction = 0.02
    system_max_width_fraction: Fraction = 0.62
    system_center_tolerance: Fraction = 0.12
    """How close to centred a narrow blob must be to count as a separator."""

    @field_validator("close_kernel", "open_kernel")
    @classmethod
    def _odd_positive(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w < 1 or h < 1:
            raise ValueError("kernel sizes must be >= 1")
        return (w, h)


class TimestampConfig(_Base):
    time_patterns: list[str] = Field(
        default_factory=lambda: [
            r"(?P<h>\d{1,2})[:.](?P<m>\d{2})\s*(?P<ampm>[APap]\.?[Mm]\.?)?",
        ]
    )
    """Regexes with ``h``/``m``/optional ``ampm`` groups, matched at the tail."""
    date_separator_patterns: list[str] = Field(
        default_factory=lambda: [
            r"^(today|yesterday)$",
            r"^\w+,?\s+\d{1,2}\s+\w+(\s+\d{4})?$",
            r"^\d{1,2}[/.]\d{1,2}[/.]\d{2,4}$",
        ]
    )
    locale: str = "en"
    dayfirst: bool = True
    position: Literal["bottom_right", "bottom_left", "outside", "none"] = "bottom_right"
    region_width_fraction: Fraction = 0.42
    region_height_fraction: Fraction = 0.45
    twelve_hour: bool = False


class SenderConfig(_Base):
    group_chat: bool = False
    me: str = "Me"
    """Name used for right-side (outgoing) bubbles."""
    other: str = "Them"
    """Name used for left-side bubbles in a 1:1 chat."""
    sender_in_bubble: bool = True
    """Group chats print the sender as a coloured first line in the bubble."""
    sender_line_max_words: int = Field(default=5, ge=1)
    known_senders: list[str] = Field(default_factory=list)
    """Helps snap OCR'd names to a consistent spelling."""


class FingerprintConfig(_Base):
    footer_placeholders: list[str] = Field(default_factory=list)
    """Composer hint text, e.g. "Type a message" — a strong app signal."""
    header_keywords: list[str] = Field(default_factory=list)
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "background": 1.0,
            "outgoing": 1.4,
            "incoming": 1.0,
            "theme": 0.6,
            "footer_text": 1.6,
        }
    )
    min_score: float = Field(default=0.45, ge=0.0, le=1.0)
    """Below this, auto-detection reports no confident match."""


class Profile(_Base):
    """Everything app- and theme-specific, in one validated object."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    display_name: str = ""
    app: str = ""
    theme: Theme = "auto"
    version: int = 1
    notes: str = ""
    layout: LayoutConfig = Field(default_factory=LayoutConfig)
    colors: ColorConfig = Field(default_factory=ColorConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    timestamps: TimestampConfig = Field(default_factory=TimestampConfig)
    senders: SenderConfig = Field(default_factory=SenderConfig)
    fingerprint: FingerprintConfig = Field(default_factory=FingerprintConfig)
    emoji_policy: Literal["placeholder", "vision", "drop"] = "placeholder"
    emoji_placeholder: str = "[emoji]"
    source_path: str = ""
    """Where this profile was loaded from; set by the loader, not the YAML."""

    @model_validator(mode="after")
    def _check_margins(self) -> Profile:
        lay = self.layout
        if lay.header_fraction + lay.footer_fraction >= 1.0:
            raise ValueError("header_fraction + footer_fraction must leave a chat area")
        if lay.left_margin_fraction + lay.right_margin_fraction >= 1.0:
            raise ValueError("left and right margins must leave a chat area")
        return self

    @field_validator("display_name")
    @classmethod
    def _default_display(cls, v: str) -> str:
        return v

    def describe(self) -> str:
        return self.display_name or self.name

    def dump_yaml(self) -> str:
        import yaml

        data: dict[str, Any] = self.model_dump(mode="json", exclude={"source_path"})
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
