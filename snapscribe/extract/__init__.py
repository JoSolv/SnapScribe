"""Text extraction: classical OCR and an optional vision model, one interface."""

from snapscribe.extract.base import (
    BaseExtractor,
    Extractor,
    ExtractResult,
    NullExtractor,
    TextLine,
    available_engines,
    get_extractor,
    iter_available,
    register,
)
from snapscribe.extract.emoji import emoji_boxes, insert_placeholders
from snapscribe.extract.ocr import OcrExtractor
from snapscribe.extract.preprocess import prepare_crop
from snapscribe.extract.stage import ExtractStage
from snapscribe.extract.vision import SUGGESTED_MODELS, VisionExtractor

__all__ = [
    "SUGGESTED_MODELS",
    "BaseExtractor",
    "ExtractResult",
    "ExtractStage",
    "Extractor",
    "NullExtractor",
    "OcrExtractor",
    "TextLine",
    "VisionExtractor",
    "available_engines",
    "emoji_boxes",
    "get_extractor",
    "insert_placeholders",
    "iter_available",
    "prepare_crop",
    "register",
]
