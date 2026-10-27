"""Render chat screenshots from an export, with matching ground truth.

A real corpus needs a phone. This renderer stands in until you have one: it
draws a plausible WhatsApp-style UI from an export file, so every pixel's
provenance is known and the ground truth is exact. It deliberately includes the
things that break naive pipelines — bubbles clipped by the viewport edge, a
status-bar clock that changes between screenshots, colour emoji, media bubbles,
date separators and a day boundary.

It is **not** a substitute for real screenshots: it cannot reproduce font
rendering, wallpapers, compression artefacts or a specific app version. Treat
good scores here as "the plumbing works", and real screenshots as the
measurement that counts.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from snapscribe.core.models import AttachmentType
from snapscribe.corpus.whatsapp_export import ExportMessage, infer_owner, parse_export_file
from snapscribe.imaging import hex_to_bgr
from snapscribe.profiles import Profile, load_profile

FONT_DIRS = (
    Path("/usr/share/fonts/opentype/inter"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/dejavu"),
)
EMOJI_FONT = Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf")
EMOJI_NATIVE_SIZE = 109
"""NotoColorEmoji is a bitmap font and only renders at this size."""

_EMOJI_RE = re.compile(
    "(?:[\U0001f000-\U0001faff☀-➿⬀-⯿←-⇿⬅-⬇〰〽㊗㊙©®][︀-️‍\U0001f3fb-\U0001f3ff]*)+"
)

SENDER_COLORS = ("#E542A3", "#1F9BCF", "#C4820E", "#6E5BC8", "#0E8F6F", "#D0574F")


def _rgb(hex_color: str) -> tuple[int, int, int]:
    b, g, r = hex_to_bgr(hex_color)
    return (r, g, b)


def _mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _find_font(*names: str) -> Path:
    for name in names:
        for d in FONT_DIRS:
            p = d / name
            if p.exists():
                return p
    raise FileNotFoundError(f"none of {names} found in {[str(d) for d in FONT_DIRS]}")


@dataclass
class Palette:
    """Screen colours, derived from the profile so the corpus matches it."""

    background: tuple[int, int, int]
    incoming: tuple[int, int, int]
    outgoing: tuple[int, int, int]
    system: tuple[int, int, int]
    text: tuple[int, int, int]
    subtext: tuple[int, int, int]
    header_bg: tuple[int, int, int]
    header_text: tuple[int, int, int]
    composer_bg: tuple[int, int, int]
    footer_bg: tuple[int, int, int]
    accent: tuple[int, int, int]
    tick: tuple[int, int, int]
    dark: bool

    @classmethod
    def from_profile(cls, profile: Profile) -> Palette:
        dark = profile.theme == "dark"
        bg = _rgb(profile.colors.background or ("#0B141A" if dark else "#EFEAE2"))
        inc = _rgb(profile.colors.incoming_bubble or ("#202C33" if dark else "#FFFFFF"))
        out = _rgb(profile.colors.outgoing_bubble or ("#005C4B" if dark else "#D9FDD3"))
        sysc = _rgb(profile.colors.system_bubble or ("#182229" if dark else "#FFFFFF"))
        txt = _rgb(profile.colors.text or ("#E9EDEF" if dark else "#111B21"))
        return cls(
            background=bg,
            incoming=inc,
            outgoing=out,
            system=sysc,
            text=txt,
            subtext=_mix(txt, bg, 0.45),
            header_bg=_rgb("#202C33") if dark else _rgb("#F0F2F5"),
            header_text=_rgb("#E9EDEF") if dark else _rgb("#111B21"),
            composer_bg=_rgb("#2A3942") if dark else (255, 255, 255),
            footer_bg=_rgb("#0B141A") if dark else _rgb("#F0F2F5"),
            accent=_rgb("#00A884"),
            tick=_rgb("#53BDEB"),
            dark=dark,
        )


@dataclass
class RenderSpec:
    """Device geometry and type sizes, in pixels."""

    width: int = 1080
    height: int = 2340
    status_h: int = 64
    appbar_h: int = 136
    composer_h: int = 160
    body_px: int = 40
    sender_px: int = 36
    meta_px: int = 27
    header_px: int = 44
    radius: int = 24
    pad_x: int = 26
    pad_y: int = 18
    gap: int = 18
    group_gap: int = 30
    max_bubble_fraction: float = 0.78
    avatar_px: int = 0
    """Set > 0 to draw an avatar gutter for incoming group messages."""
    media_w: int = 560
    media_h: int = 420
    overlap_px: int = 260
    jitter: float = 0.0
    """0..1 colour jitter, so the corpus is not a pixel-exact copy of the profile."""

    @property
    def header_h(self) -> int:
        return self.status_h + self.appbar_h

    @property
    def footer_h(self) -> int:
        return self.composer_h

    @property
    def chat_h(self) -> int:
        return self.height - self.header_h - self.footer_h


@dataclass
class RenderedMessage:
    """Where a ground-truth message ended up on the tall canvas."""

    gt_index: int
    y: int
    h: int
    x: int
    w: int
    side: str
    sender: str
    body: str
    timestamp: dt.datetime | None
    attachment_type: str
    is_system: bool


@dataclass
class RenderResult:
    directory: Path
    images: list[Path] = field(default_factory=list)
    ground_truth_csv: Path | None = None
    meta_json: Path | None = None
    messages: list[RenderedMessage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _Fonts:
    def __init__(self, spec: RenderSpec) -> None:
        regular = _find_font("Inter-Regular.otf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf")
        medium = _find_font("Inter-Medium.otf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf")
        semibold = _find_font(
            "Inter-SemiBold.otf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"
        )
        self.body = ImageFont.truetype(str(regular), spec.body_px)
        self.sender = ImageFont.truetype(str(semibold), spec.sender_px)
        self.meta = ImageFont.truetype(str(regular), spec.meta_px)
        self.header = ImageFont.truetype(str(medium), spec.header_px)
        self.status = ImageFont.truetype(str(medium), 34)
        self.pill = ImageFont.truetype(str(medium), 30)
        self._emoji_cache: dict[tuple[str, int], Image.Image] = {}

    def emoji(self, cluster: str, size: int) -> Image.Image | None:
        key = (cluster, size)
        if key in self._emoji_cache:
            return self._emoji_cache[key]
        if not EMOJI_FONT.exists():
            return None
        try:
            font = ImageFont.truetype(str(EMOJI_FONT), EMOJI_NATIVE_SIZE)
        except OSError:  # pragma: no cover
            return None
        canvas = Image.new("RGBA", (EMOJI_NATIVE_SIZE * 3, EMOJI_NATIVE_SIZE * 2), (0, 0, 0, 0))
        d = ImageDraw.Draw(canvas)
        try:
            d.text((4, 4), cluster, font=font, embedded_color=True)
        except OSError:  # pragma: no cover
            return None
        bbox = canvas.getbbox()
        if bbox is None:
            return None
        glyph = canvas.crop(bbox)
        scale = size / max(glyph.height, 1)
        target = (max(1, int(glyph.width * scale)), max(1, size))
        out = glyph.resize(target, Image.LANCZOS)
        self._emoji_cache[key] = out
        return out


@dataclass
class _Token:
    text: str
    is_emoji: bool
    width: int


class ChatRenderer:
    """Draws an export as a scrollable chat screen."""

    def __init__(
        self,
        profile: Profile,
        spec: RenderSpec | None = None,
        title: str = "Chat",
        owner: str = "Me",
        seed: int = 7,
    ) -> None:
        self.profile = profile
        self.spec = spec or RenderSpec()
        self.palette = Palette.from_profile(profile)
        self.fonts = _Fonts(self.spec)
        self.title = title
        self.owner = owner
        self.rng = random.Random(seed)
        self._sender_color: dict[str, tuple[int, int, int]] = {}

    # ------------------------------------------------------------- utilities

    def _jitter(self, color: tuple[int, int, int]) -> tuple[int, int, int]:
        amount = self.spec.jitter
        if amount <= 0:
            return color
        span = round(10 * amount)
        return tuple(  # type: ignore[return-value]
            max(0, min(255, c + self.rng.randint(-span, span))) for c in color
        )

    def sender_color(self, name: str) -> tuple[int, int, int]:
        if name not in self._sender_color:
            idx = len(self._sender_color) % len(SENDER_COLORS)
            self._sender_color[name] = _rgb(SENDER_COLORS[idx])
        return self._sender_color[name]

    def _tokenize(self, text: str, size: int) -> list[_Token]:
        tokens: list[_Token] = []
        measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        for part in re.split(r"(\s+)", text):
            if not part:
                continue
            if part.isspace():
                tokens.append(
                    _Token(" ", False, int(measure.textlength(" ", font=self.fonts.body)))
                )
                continue
            pos = 0
            for m in _EMOJI_RE.finditer(part):
                if m.start() > pos:
                    chunk = part[pos : m.start()]
                    tokens.append(
                        _Token(chunk, False, int(measure.textlength(chunk, font=self.fonts.body)))
                    )
                cluster = m.group(0)
                glyph = self.fonts.emoji(cluster, size)
                width = (
                    glyph.width if glyph else int(measure.textlength(cluster, font=self.fonts.body))
                )
                tokens.append(_Token(cluster, True, width + 2))
                pos = m.end()
            if pos < len(part):
                chunk = part[pos:]
                tokens.append(
                    _Token(chunk, False, int(measure.textlength(chunk, font=self.fonts.body)))
                )
        return tokens

    def _wrap(self, text: str, max_width: int) -> tuple[list[list[_Token]], int]:
        lines: list[list[_Token]] = []
        longest = 0
        for paragraph in text.split("\n"):
            tokens = self._tokenize(paragraph, self.spec.body_px)
            line: list[_Token] = []
            width = 0
            for tok in tokens:
                if tok.text == " " and not line:
                    continue
                if width + tok.width > max_width and line:
                    lines.append(line)
                    longest = max(longest, width)
                    line, width = (
                        ([] if tok.text == " " else [tok]),
                        (0 if tok.text == " " else tok.width),
                    )
                    continue
                line.append(tok)
                width += tok.width
            lines.append(line)
            longest = max(longest, width)
        return lines, longest

    def _draw_text_line(
        self,
        draw: ImageDraw.ImageDraw,
        canvas: Image.Image,
        tokens: list[_Token],
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        cursor = x
        for tok in tokens:
            if tok.is_emoji:
                glyph = self.fonts.emoji(tok.text, self.spec.body_px)
                if glyph is not None:
                    canvas.paste(glyph, (cursor, y + 2), glyph)
                else:
                    draw.text((cursor, y), tok.text, font=self.fonts.body, fill=color)
            else:
                draw.text((cursor, y), tok.text, font=self.fonts.body, fill=color)
            cursor += tok.width

    # ---------------------------------------------------------------- chrome

    def draw_header(self, canvas: Image.Image, clock: str, subtitle: str = "online") -> None:
        spec, pal = self.spec, self.palette
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, spec.width, spec.status_h], fill=pal.header_bg)
        d.text((36, 16), clock, font=self.fonts.status, fill=pal.header_text)
        # Signal/wifi/battery, drawn as simple shapes: stable across screenshots.
        right = spec.width - 40
        for i in range(4):
            h = 10 + i * 8
            d.rectangle(
                [right - 90 + i * 16, 42 - h, right - 82 + i * 16, 42], fill=pal.header_text
            )
        d.rounded_rectangle(
            [right - 44, 20, right - 4, 44], radius=6, outline=pal.header_text, width=3
        )
        d.rectangle([right - 41, 24, right - 20, 40], fill=pal.header_text)

        top = spec.status_h
        d.rectangle([0, top, spec.width, top + spec.appbar_h], fill=pal.header_bg)
        # back arrow
        cx, cy = 48, top + spec.appbar_h // 2
        d.line(
            [(cx + 20, cy - 22), (cx - 2, cy), (cx + 20, cy + 22)], fill=pal.header_text, width=6
        )
        # avatar
        ax = 104
        d.ellipse([ax, cy - 42, ax + 84, cy + 42], fill=_mix(pal.header_bg, pal.accent, 0.5))
        d.text((ax + 240, cy - 44), self.title, font=self.fonts.header, fill=pal.header_text)
        d.text(
            (ax + 240, cy + 10),
            subtitle,
            font=self.fonts.meta,
            fill=_mix(pal.header_text, pal.header_bg, 0.4),
        )
        for i in range(3):  # overflow dots
            x = spec.width - 44
            d.ellipse([x - 4, cy - 24 + i * 18, x + 4, cy - 16 + i * 18], fill=pal.header_text)

    def draw_footer(self, canvas: Image.Image, placeholder: str = "Type a message") -> None:
        spec, pal = self.spec, self.palette
        d = ImageDraw.Draw(canvas)
        top = spec.height - spec.footer_h
        d.rectangle([0, top, spec.width, spec.height], fill=pal.footer_bg)
        box = [24, top + 24, spec.width - 150, top + spec.composer_h - 24]
        d.rounded_rectangle(box, radius=(box[3] - box[1]) // 2, fill=pal.composer_bg)
        d.text((box[0] + 100, top + 52), placeholder, font=self.fonts.header, fill=pal.subtext)
        d.ellipse([box[0] + 26, top + 58, box[0] + 66, top + 98], outline=pal.subtext, width=4)
        send = spec.width - 116
        d.ellipse([send, top + 30, send + 92, top + 122], fill=pal.accent)
        d.polygon(
            [(send + 34, top + 54), (send + 66, top + 76), (send + 34, top + 98)],
            fill=(255, 255, 255),
        )

    # --------------------------------------------------------------- bubbles

    def _fake_photo(self, w: int, h: int, kind: str) -> Image.Image:
        """A synthetic "photo": gradient plus shapes, so media bubbles have the
        high colour variance that real photos do."""
        img = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(img)
        c1 = (self.rng.randint(30, 200), self.rng.randint(30, 200), self.rng.randint(30, 200))
        c2 = (self.rng.randint(30, 200), self.rng.randint(30, 200), self.rng.randint(30, 200))
        for y in range(h):
            d.line([(0, y), (w, y)], fill=_mix(c1, c2, y / max(1, h - 1)))
        for _ in range(7):
            x0, y0 = self.rng.randint(0, w - 40), self.rng.randint(0, h - 40)
            r = self.rng.randint(20, max(21, min(w, h) // 3))
            shade = (self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255))
            d.ellipse([x0, y0, x0 + r, y0 + r], fill=shade)
        img = img.filter(ImageFilter.GaussianBlur(1.2))
        d = ImageDraw.Draw(img)
        if kind == "video":
            cx, cy = w // 2, h // 2
            d.ellipse([cx - 56, cy - 56, cx + 56, cy + 56], fill=(0, 0, 0))
            d.polygon([(cx - 18, cy - 28), (cx + 30, cy), (cx - 18, cy + 28)], fill=(255, 255, 255))
            d.text((16, h - 44), "0:24", font=self.fonts.meta, fill=(255, 255, 255))
        return img

    def _message_size(
        self, msg: ExportMessage, side: str, show_sender: bool
    ) -> tuple[int, int, list[list[_Token]]]:
        spec = self.spec
        max_text = int(spec.width * spec.max_bubble_fraction) - 2 * spec.pad_x
        attachment = msg.attachment_type
        body = msg.body
        lines: list[list[_Token]] = []
        text_w = 0
        text_h = 0
        if body:
            lines, text_w = self._wrap(body, max_text)
            text_h = len(lines) * int(spec.body_px * 1.34)

        media_h = 0
        media_w = 0
        if attachment in (AttachmentType.IMAGE, AttachmentType.VIDEO, AttachmentType.GIF):
            media_w, media_h = spec.media_w, spec.media_h
        elif attachment is AttachmentType.STICKER:
            media_w, media_h = 320, 320
        elif attachment is AttachmentType.AUDIO:
            media_w, media_h = 480, 90
        elif attachment in (
            AttachmentType.DOCUMENT,
            AttachmentType.CONTACT,
            AttachmentType.LOCATION,
        ):
            media_w, media_h = 520, 130
        elif attachment is AttachmentType.DELETED:
            lines, text_w = self._wrap("This message was deleted", max_text)
            text_h = len(lines) * int(spec.body_px * 1.34)

        sender_h = int(spec.sender_px * 1.5) if show_sender else 0
        meta_h = int(spec.meta_px * 1.5)
        inner_w = max(text_w, media_w, 220)
        w = min(int(spec.width * spec.max_bubble_fraction), inner_w + 2 * spec.pad_x)
        h = (
            spec.pad_y * 2
            + sender_h
            + media_h
            + (12 if media_h and text_h else 0)
            + text_h
            + meta_h
        )
        return w, h, lines

    def draw_bubble(
        self,
        canvas: Image.Image,
        msg: ExportMessage,
        side: str,
        x: int,
        y: int,
        w: int,
        h: int,
        lines: list[list[_Token]],
        show_sender: bool,
    ) -> None:
        spec, pal = self.spec, self.palette
        d = ImageDraw.Draw(canvas)
        fill = self._jitter(pal.outgoing if side == "right" else pal.incoming)
        d.rounded_rectangle([x, y, x + w, y + h], radius=spec.radius, fill=fill)
        # tail
        if side == "right":
            d.polygon([(x + w - 6, y + 4), (x + w + 14, y + 2), (x + w - 6, y + 34)], fill=fill)
        else:
            d.polygon([(x + 6, y + 4), (x - 14, y + 2), (x + 6, y + 34)], fill=fill)

        cursor_y = y + spec.pad_y
        if show_sender:
            d.text(
                (x + spec.pad_x, cursor_y),
                msg.sender,
                font=self.fonts.sender,
                fill=self.sender_color(msg.sender),
            )
            cursor_y += int(spec.sender_px * 1.5)

        att = msg.attachment_type
        if att in (
            AttachmentType.IMAGE,
            AttachmentType.VIDEO,
            AttachmentType.GIF,
            AttachmentType.STICKER,
        ):
            mw = 320 if att is AttachmentType.STICKER else spec.media_w
            mh = 320 if att is AttachmentType.STICKER else spec.media_h
            photo = self._fake_photo(mw, mh, "video" if att is AttachmentType.VIDEO else "image")
            rounded = Image.new("L", (mw, mh), 0)
            ImageDraw.Draw(rounded).rounded_rectangle([0, 0, mw, mh], radius=16, fill=255)
            canvas.paste(photo, (x + spec.pad_x, cursor_y), rounded)
            cursor_y += mh + (12 if lines else 0)
        elif att is AttachmentType.AUDIO:
            d.ellipse(
                [x + spec.pad_x, cursor_y, x + spec.pad_x + 72, cursor_y + 72], fill=pal.accent
            )
            d.polygon(
                [
                    (x + spec.pad_x + 28, cursor_y + 20),
                    (x + spec.pad_x + 54, cursor_y + 36),
                    (x + spec.pad_x + 28, cursor_y + 52),
                ],
                fill=(255, 255, 255),
            )
            bar_x = x + spec.pad_x + 96
            for i in range(26):
                bh = 6 + int(22 * abs(math.sin(i * 0.9)))
                d.rectangle(
                    [
                        bar_x + i * 12,
                        cursor_y + 36 - bh // 2,
                        bar_x + i * 12 + 5,
                        cursor_y + 36 + bh // 2,
                    ],
                    fill=pal.subtext,
                )
            d.text((bar_x, cursor_y + 56), "0:42", font=self.fonts.meta, fill=pal.subtext)
            cursor_y += 90 + (12 if lines else 0)
        elif att in (AttachmentType.DOCUMENT, AttachmentType.CONTACT, AttachmentType.LOCATION):
            d.rounded_rectangle(
                [x + spec.pad_x, cursor_y, x + w - spec.pad_x, cursor_y + 110],
                radius=12,
                fill=_mix(fill, pal.text, 0.16),
            )
            d.rectangle(
                [x + spec.pad_x + 18, cursor_y + 20, x + spec.pad_x + 78, cursor_y + 92],
                fill=pal.subtext,
            )
            label = {
                "document": "outline-v3.pdf",
                "contact": "Contact card",
                "location": "Shared location",
            }[att.value]
            d.text(
                (x + spec.pad_x + 100, cursor_y + 38), label, font=self.fonts.meta, fill=pal.text
            )
            cursor_y += 130 + (12 if lines else 0)

        line_h = int(spec.body_px * 1.34)
        for line in lines:
            self._draw_text_line(d, canvas, line, x + spec.pad_x, cursor_y, pal.text)
            cursor_y += line_h

        # timestamp + ticks, bottom-right inside the bubble
        stamp = msg.timestamp.strftime("%H:%M") if msg.timestamp else ""
        if stamp:
            tw = int(d.textlength(stamp, font=self.fonts.meta))
            tx = x + w - spec.pad_x - tw - (46 if side == "right" else 0)
            ty = y + h - spec.pad_y - spec.meta_px - 2
            d.text((tx, ty), stamp, font=self.fonts.meta, fill=pal.subtext)
            if side == "right":
                bx = x + w - spec.pad_x - 40
                for off in (0, 12):
                    d.line(
                        [(bx + off, ty + 16), (bx + off + 8, ty + 24), (bx + off + 22, ty + 4)],
                        fill=pal.tick,
                        width=4,
                    )

    def draw_separator(self, canvas: Image.Image, label: str, y: int) -> int:
        spec, pal = self.spec, self.palette
        d = ImageDraw.Draw(canvas)
        tw = int(d.textlength(label, font=self.fonts.pill))
        w = tw + 60
        x = (spec.width - w) // 2
        h = 64
        d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=self._jitter(pal.system))
        d.text((x + 30, y + 16), label, font=self.fonts.pill, fill=pal.subtext)
        return h

    def draw_system(self, canvas: Image.Image, text: str, y: int) -> int:
        spec, pal = self.spec, self.palette
        d = ImageDraw.Draw(canvas)
        max_text = int(spec.width * 0.74)
        lines, tw = self._wrap(text, max_text)
        line_h = int(spec.body_px * 1.34)
        h = len(lines) * line_h + 2 * spec.pad_y
        w = min(max_text + 2 * spec.pad_x, tw + 2 * spec.pad_x)
        x = (spec.width - w) // 2
        d.rounded_rectangle([x, y, x + w, y + h], radius=18, fill=self._jitter(pal.system))
        cy = y + spec.pad_y
        for line in lines:
            self._draw_text_line(d, canvas, line, x + spec.pad_x, cy, pal.subtext)
            cy += line_h
        return h

    # ----------------------------------------------------------- composition

    def layout_conversation(
        self, messages: list[ExportMessage], group_chat: bool
    ) -> tuple[Image.Image, list[RenderedMessage]]:
        """Draw the whole conversation onto one tall canvas.

        Returns the canvas and where each ground-truth message landed, which is
        what makes per-screenshot visibility (and therefore honest recall
        measurement) computable.
        """
        spec, pal = self.spec, self.palette
        # Two passes: measure, then draw onto a canvas of the right height.
        plan: list[
            tuple[int, ExportMessage, str, int, int, int, list[list[_Token]], bool, str]
        ] = []
        y = spec.gap
        last_sender: str | None = None
        last_date: dt.date | None = None

        for i, msg in enumerate(messages):
            side = "right" if msg.sender == self.owner else "left"
            msg_date = msg.timestamp.date() if msg.timestamp else None
            if msg_date is not None and msg_date != last_date:
                label = _date_label(msg_date, messages)
                plan.append((i, msg, "separator", 0, y, 64, [], False, label))
                y += 64 + spec.gap
                last_date = msg_date
                last_sender = None

            if msg.is_system:
                lines, _tw = self._wrap(msg.body, int(spec.width * 0.74))
                h = len(lines) * int(spec.body_px * 1.34) + 2 * spec.pad_y
                plan.append((i, msg, "system", 0, y, h, lines, False, ""))
                y += h + spec.group_gap
                last_sender = None
                continue

            show_sender = bool(group_chat and side == "left" and msg.sender != last_sender)
            w, h, lines = self._message_size(msg, side, show_sender)
            gutter = spec.avatar_px + 20 if (spec.avatar_px and side == "left") else 0
            x = spec.width - w - 24 if side == "right" else 24 + gutter
            same_sender = msg.sender == last_sender
            y += 0 if same_sender else (spec.group_gap - spec.gap)
            plan.append((i, msg, "bubble", x, y, h, lines, show_sender, side))
            y += h + spec.gap
            last_sender = msg.sender

        total_h = y + spec.gap
        canvas = Image.new("RGB", (spec.width, total_h), self._jitter(pal.background))
        rendered: list[RenderedMessage] = []

        for i, msg, kind, x, top, h, lines, show_sender, extra in plan:
            if kind == "separator":
                self.draw_separator(canvas, extra, top)
                continue
            if kind == "system":
                self.draw_system(canvas, msg.body, top)
                rendered.append(
                    RenderedMessage(
                        gt_index=i,
                        y=top,
                        h=h,
                        x=int(spec.width * 0.13),
                        w=int(spec.width * 0.74),
                        side="center",
                        sender="",
                        body=msg.body,
                        timestamp=msg.timestamp,
                        attachment_type=msg.attachment_type.value,
                        is_system=True,
                    )
                )
                continue
            side = extra
            w, _, _ = self._message_size(msg, side, show_sender)
            if spec.avatar_px and side == "left" and show_sender:
                d = ImageDraw.Draw(canvas)
                d.ellipse(
                    [24, top + h - spec.avatar_px, 24 + spec.avatar_px, top + h],
                    fill=_mix(pal.background, self.sender_color(msg.sender), 0.7),
                )
            self.draw_bubble(canvas, msg, side, x, top, w, h, lines, show_sender)
            rendered.append(
                RenderedMessage(
                    gt_index=i,
                    y=top,
                    h=h,
                    x=x,
                    w=w,
                    side=side,
                    sender=msg.sender,
                    body=msg.body,
                    timestamp=msg.timestamp,
                    attachment_type=msg.attachment_type.value,
                    is_system=False,
                )
            )
        return canvas, rendered

    def windows(self, content_h: int, messages: list[RenderedMessage]) -> list[tuple[int, int]]:
        """Viewport offsets into the tall canvas.

        The overlap is forced to exceed the tallest message, which is what
        guarantees every message is fully visible in at least one screenshot —
        otherwise a tall media bubble could be clipped in every single one and
        no pipeline could ever recover it.
        """
        spec = self.spec
        chat_h = spec.chat_h
        tallest = max((m.h for m in messages), default=0)
        overlap = max(spec.overlap_px, tallest + 2 * spec.gap)
        if overlap >= chat_h:
            overlap = chat_h - spec.gap
        step = max(1, chat_h - overlap)
        offsets: list[tuple[int, int]] = []
        y = 0
        while True:
            offsets.append((y, min(content_h, y + chat_h)))
            if y + chat_h >= content_h:
                break
            y += step
        return offsets


def _date_label(day: dt.date, messages: list[ExportMessage]) -> str:
    """WhatsApp shows TODAY/YESTERDAY relative to *now*; the corpus is fixed in
    time, so relative labels are generated relative to the last message's day —
    that way "Yesterday"/"Today" appear and the date resolver gets exercised."""
    stamps = [m.timestamp.date() for m in messages if m.timestamp]
    if not stamps:
        return day.strftime("%d/%m/%Y")
    anchor = max(stamps)
    delta = (anchor - day).days
    if delta == 0:
        return "TODAY"
    if delta == 1:
        return "YESTERDAY"
    return day.strftime("%A, %d %B %Y").upper()


def build_corpus(
    export_path: str | Path,
    out_dir: str | Path,
    profile_name: str = "whatsapp_light",
    title: str = "",
    owner: str = "",
    group_chat: bool | None = None,
    spec: RenderSpec | None = None,
    seed: int = 7,
    prefix: str = "",
) -> RenderResult:
    """Render an export into screenshots plus ``ground_truth.csv``."""
    export_path = Path(export_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.png"):
        stale.unlink()

    profile = load_profile(profile_name)
    messages = parse_export_file(export_path)
    if not messages:
        raise ValueError(f"{export_path} contained no parseable messages")
    owner = owner or infer_owner(messages) or "Me"
    senders = {m.sender for m in messages if m.sender}
    if group_chat is None:
        group_chat = len(senders) > 2
    title = title or (
        export_path.stem.replace("_", " ").title() if group_chat else _other(senders, owner)
    )

    renderer = ChatRenderer(profile, spec=spec, title=title, owner=owner, seed=seed)
    canvas, rendered = renderer.layout_conversation(messages, group_chat=group_chat)
    offsets = renderer.windows(canvas.height, rendered)

    rspec = renderer.spec
    # Screenshots are taken *after* the conversation happened, so the capture
    # time is anchored to the last message. Getting this wrong would make
    # "TODAY"/"YESTERDAY" resolve against the wrong day — the same failure a
    # user hits when they transcribe screenshots months later without passing
    # --anchor-date.
    last_message = max((m.timestamp for m in messages if m.timestamp), default=None)
    capture_base = (
        dt.datetime.combine(last_message.date(), dt.time(21, 30))
        if last_message
        else dt.datetime(2024, 1, 2, 10, 0)
    )
    base_clock = capture_base
    images: list[Path] = []
    visibility: list[dict[str, object]] = []
    stem = prefix or export_path.stem

    for n, (y0, y1) in enumerate(offsets):
        frame = Image.new("RGB", (rspec.width, rspec.height), renderer.palette.background)
        view = canvas.crop((0, y0, rspec.width, y1))
        frame.paste(view, (0, rspec.header_h))
        if view.height < rspec.chat_h:  # last window: pad with background
            pad = Image.new(
                "RGB",
                (rspec.width, rspec.chat_h - view.height),
                renderer.palette.background,
            )
            frame.paste(pad, (0, rspec.header_h + view.height))
        clock = (base_clock + dt.timedelta(minutes=n * 3)).strftime("%H:%M")
        renderer.draw_header(frame, clock=clock)
        renderer.draw_footer(frame)

        shot_time = capture_base + dt.timedelta(minutes=n)
        name = f"{stem}_{profile.theme}_{shot_time:%Y%m%d-%H%M%S}_{n:03d}.png"
        path = out / name
        frame.save(path)
        images.append(path)

        fully_visible = [m.gt_index for m in rendered if m.y >= y0 and m.y + m.h <= y1]
        clipped = [
            m.gt_index
            for m in rendered
            if not (m.y >= y0 and m.y + m.h <= y1) and m.y < y1 and m.y + m.h > y0
        ]
        visibility.append(
            {"image": name, "offset": y0, "fully_visible": fully_visible, "clipped": clipped}
        )

    gt_csv = out / "ground_truth.csv"
    with gt_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["index", "sender", "side", "body", "timestamp", "attachment_type", "is_system"]
        )
        for m in rendered:
            writer.writerow(
                [
                    m.gt_index,
                    m.sender,
                    m.side,
                    m.body,
                    m.timestamp.isoformat(sep=" ") if m.timestamp else "",
                    m.attachment_type,
                    "true" if m.is_system else "false",
                ]
            )

    covered = {i for v in visibility for i in v["fully_visible"]}  # type: ignore[union-attr]
    warnings: list[str] = []
    missing = [m.gt_index for m in rendered if m.gt_index not in covered]
    if missing:
        warnings.append(
            f"{len(missing)} message(s) are clipped in every screenshot (indices {missing}); "
            "raise overlap_px or the viewport height"
        )

    meta = {
        "source_export": str(export_path),
        "profile": profile.name,
        "theme": profile.theme,
        "title": title,
        "owner": owner,
        "group_chat": group_chat,
        "device": {
            "width": rspec.width,
            "height": rspec.height,
            "header_h": rspec.header_h,
            "footer_h": rspec.footer_h,
        },
        "content_height": canvas.height,
        "message_count": len(rendered),
        "images": [p.name for p in images],
        "visibility": visibility,
        "warnings": warnings,
        "generated_by": "snapscribe.corpus.render",
    }
    meta_path = out / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    export_copy = out / "export.txt"
    export_copy.write_text(export_path.read_text(encoding="utf-8-sig"), encoding="utf-8")

    return RenderResult(
        directory=out,
        images=images,
        ground_truth_csv=gt_csv,
        meta_json=meta_path,
        messages=rendered,
        warnings=warnings,
    )


def _other(senders: set[str], owner: str) -> str:
    rest = sorted(s for s in senders if s != owner)
    return rest[0] if rest else "Chat"
