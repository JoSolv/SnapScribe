"""App/theme profiles: YAML in, validated geometry and colours out."""

from snapscribe.profiles.fingerprint import (
    ProfileMatch,
    detect_profile,
    rank_profiles,
    score_profile,
)
from snapscribe.profiles.loader import (
    BUILTIN_DIR,
    USER_DIR,
    ProfileError,
    apply_overrides,
    list_profiles,
    load_profile,
    load_profile_file,
    parse_override,
    parse_profile,
)
from snapscribe.profiles.schema import Profile

__all__ = [
    "BUILTIN_DIR",
    "USER_DIR",
    "Profile",
    "ProfileError",
    "ProfileMatch",
    "apply_overrides",
    "detect_profile",
    "list_profiles",
    "load_profile",
    "load_profile_file",
    "parse_override",
    "parse_profile",
    "rank_profiles",
    "score_profile",
]
