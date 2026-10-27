"""Find the chat area: strip the app chrome, check image quality."""

from snapscribe.normalize.chrome import ChromeDetection, detect_chrome, detect_chrome_group
from snapscribe.normalize.quality import assess_quality
from snapscribe.normalize.stage import NormalizeStage

__all__ = [
    "ChromeDetection",
    "NormalizeStage",
    "assess_quality",
    "detect_chrome",
    "detect_chrome_group",
]
