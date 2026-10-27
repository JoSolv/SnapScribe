"""Cross-check the two extraction paths and flag what a human should look at."""

from snapscribe.verify.compare import Comparison, compare_texts, normalize_for_compare, word_diff
from snapscribe.verify.rules import RuleResult, VerifyConfig, check_messages
from snapscribe.verify.stage import VerifyStage

__all__ = [
    "Comparison",
    "RuleResult",
    "VerifyConfig",
    "VerifyStage",
    "check_messages",
    "compare_texts",
    "normalize_for_compare",
    "word_diff",
]
