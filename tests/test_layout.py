"""Chrome detection and bubble segmentation on synthetic screens."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from snapscribe.core.models import ChromeInfo, RegionKind, Side
from snapscribe.imaging import hex_to_bgr
from snapscribe.layout.bubbles import media_rows, remove_icon_blobs, segment_chat
from snapscribe.layout.overlay import draw_overlay
from snapscribe.normalize.chrome import detect_chrome_group, row_stability
from snapscribe.normalize.quality import assess_quality, looks_rendered
from snapscribe.profiles import load_profile

BG = "#EFEAE2"
IN = "#FFFFFF"
OUT = "#D9FDD3"
HEADER = "#F0F2F5"


def _screen(bubbles, width=400, height=800, header=60, footer=50, clock="12:00"):
    """A fake chat screen: fixed chrome, bubbles at the given positions."""
    image = np.full((height, width, 3), hex_to_bgr(BG), np.uint8)
    image[:header] = hex_to_bgr(HEADER)
    image[height - footer :] = hex_to_bgr(HEADER)
    cv2.putText(image, clock, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, "Jonas", (150, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    for x, y, w, h, colour in bubbles:
        cv2.rectangle(image, (x, y), (x + w, y + h), hex_to_bgr(colour), -1)
    return image


def test_row_stability_finds_the_static_rows():
    a = _screen([(20, 100, 200, 60, IN)], clock="12:00")
    b = _screen([(20, 300, 200, 60, IN)], clock="12:31")
    stability = row_stability([a, b])
    assert stability[:5].mean() > 0.9  # top of the status bar
    assert stability[120] < 0.5  # a bubble in one image only


def test_detect_chrome_group_measures_header_and_footer():
    images = [
        _screen([(20, 100, 200, 60, IN)], clock="12:00"),
        _screen([(20, 300, 200, 60, OUT)], clock="12:31"),
        _screen([(20, 500, 200, 60, IN)], clock="13:02"),
    ]
    detection = detect_chrome_group(images, bg_tolerance=16)
    assert detection.method == "stable_rows"
    assert detection.header_height == pytest.approx(60, abs=6)
    assert detection.footer_height == pytest.approx(50, abs=6)


def test_detect_chrome_group_gives_up_on_identical_images():
    image = _screen([(20, 100, 200, 60, IN)])
    detection = detect_chrome_group([image, image.copy()])
    assert detection.method == "none"
    assert detection.notes


def _segment(bubbles, **kwargs):
    image = _screen(bubbles, **kwargs)
    chrome = ChromeInfo(image_id="img", header_height=60, footer_height=50)
    chat = chrome.chat_area(image.shape[1], image.shape[0])
    return image, segment_chat(image, chat, load_profile("whatsapp_light"), image_id="img")


def test_segmentation_finds_bubbles_and_sides():
    # Kept clear of the chat-area edges: a square bubble that ends near one is
    # treated as cut off by the viewport (see the clipping tests below).
    _, result = _segment(
        [
            (20, 160, 220, 60, IN),
            (160, 260, 220, 60, OUT),
            (20, 360, 220, 60, IN),
        ]
    )
    bubbles = [r for r in result.regions if r.kind is RegionKind.BUBBLE]
    assert len(bubbles) == 3
    assert [b.side for b in bubbles] == [Side.LEFT, Side.RIGHT, Side.LEFT]
    assert all(b.confidence >= 0.85 for b in bubbles)


def test_centred_narrow_blob_is_a_system_message():
    _, result = _segment([(150, 180, 100, 40, IN), (20, 300, 220, 60, IN)])
    kinds = [r.kind for r in result.regions if r.kind is not RegionKind.DROPPED]
    assert RegionKind.SYSTEM in kinds


def test_clipped_bubbles_are_dropped():
    _, result = _segment([(20, 62, 220, 60, IN), (20, 360, 220, 60, IN)])
    dropped = [r for r in result.regions if r.kind is RegionKind.DROPPED]
    assert any("clipped" in r.drop_reason for r in dropped)


def test_a_flat_bottom_edge_counts_as_clipped():
    """A bubble cut by the viewport ends square, even if the detected chat area
    stops a few pixels lower (dark themes, where the composer matches the
    background)."""
    image = _screen([(20, 640, 220, 100, IN)], height=800, footer=50)
    image[740:750] = hex_to_bgr(BG)  # a gap between the bubble and the chrome
    chrome = ChromeInfo(image_id="img", header_height=60, footer_height=40)
    chat = chrome.chat_area(image.shape[1], image.shape[0])
    result = segment_chat(image, chat, load_profile("whatsapp_light"), image_id="img")
    assert any("clipped at the bottom" in r.drop_reason for r in result.regions)


def test_nested_regions_are_dropped():
    """A card inside a bubble can punch a hole in the mask, leaving whatever is
    drawn inside it as its own component. It is never a separate message."""
    from snapscribe.core.models import Region
    from snapscribe.layout.bubbles import _drop_nested

    outer = Region(id="outer", image_id="img", kind=RegionKind.BUBBLE, x=20, y=150, w=260, h=140)
    inner = Region(id="inner", image_id="img", kind=RegionKind.BUBBLE, x=60, y=200, w=120, h=30)
    beside = Region(id="beside", image_id="img", kind=RegionKind.BUBBLE, x=20, y=400, w=200, h=60)
    regions = [outer, inner, beside]
    _drop_nested(regions)
    assert inner.kind is RegionKind.DROPPED
    assert "nested inside outer" in inner.drop_reason
    assert outer.kind is RegionKind.BUBBLE
    assert beside.kind is RegionKind.BUBBLE


def test_remove_icon_blobs_takes_avatars_not_letters():
    mask = np.zeros((120, 120), np.uint8)
    cv2.circle(mask, (30, 30), 14, 1, -1)  # avatar: solid and square-ish
    cv2.putText(mask, "W", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1, 2)  # a glyph
    cleaned, removed = remove_icon_blobs(mask, max_size=40)
    assert len(removed) == 1
    assert cleaned[60 - 20 : 60, 60:100].any()  # the glyph survived


def test_media_rows_flags_photos_but_not_emoji():
    rng = np.random.default_rng(0)
    photo = np.full((200, 300, 3), hex_to_bgr(IN), np.uint8)
    photo[40:160, 10:290] = rng.integers(0, 255, (120, 280, 3), dtype=np.uint8)
    assert media_rows(photo, color_count=9, min_rows=40)

    emoji = np.full((200, 300, 3), hex_to_bgr(IN), np.uint8)
    emoji[40:160, 10:70] = rng.integers(0, 255, (120, 60, 3), dtype=np.uint8)
    assert media_rows(emoji, color_count=9, min_rows=40) == []


def test_overlay_draws_without_touching_the_original():
    image, result = _segment([(20, 100, 220, 60, IN)])
    overlay = draw_overlay(image, result.regions)
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay, image)


def test_quality_skips_rendered_screenshots():
    """Blur and glare are properties of a photograph: a light-theme chat is
    30% pure white, which a glare detector reads as a reflection."""
    image = _screen([(20, 160, 220, 60, IN)])
    assert looks_rendered(image)
    quality = assess_quality(image)
    assert not quality.has_glare
    assert not quality.is_blurry
    assert not any("glare" in w or "blur" in w for w in quality.warnings)
