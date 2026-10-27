"""Turn a chat area into labelled regions: bubbles, sides, system messages."""

from snapscribe.layout.bubbles import SegmentationResult, segment_chat
from snapscribe.layout.overlay import draw_overlay, save_overlay
from snapscribe.layout.stage import LayoutStage

__all__ = [
    "LayoutStage",
    "SegmentationResult",
    "draw_overlay",
    "save_overlay",
    "segment_chat",
]
