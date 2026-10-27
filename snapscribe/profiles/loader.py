"""Loading, listing and overriding profiles."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from snapscribe.profiles.schema import Profile

BUILTIN_DIR = Path(__file__).parent / "builtin"
USER_DIR = Path.home() / ".config" / "snapscribe" / "profiles"


class ProfileError(ValueError):
    """A profile could not be loaded or validated."""


def _format_validation_error(source: str, exc: ValidationError) -> str:
    lines = [f"{source} is not a valid profile:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"  {loc}: {err['msg']}")
    return "\n".join(lines)


def parse_profile(data: dict[str, Any], source: str = "<dict>") -> Profile:
    if not isinstance(data, dict):
        raise ProfileError(f"{source}: expected a YAML mapping at the top level")
    try:
        profile = Profile.model_validate(data)
    except ValidationError as exc:
        raise ProfileError(_format_validation_error(source, exc)) from exc
    profile.source_path = source
    return profile


def load_profile_file(path: str | Path) -> Profile:
    p = Path(path)
    if not p.exists():
        raise ProfileError(f"no such profile file: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"{p} is not valid YAML: {exc}") from exc
    return parse_profile(data or {}, source=str(p))


def profile_search_paths() -> list[Path]:
    return [d for d in (USER_DIR, BUILTIN_DIR) if d.is_dir()]


@lru_cache(maxsize=1)
def _builtin_index() -> dict[str, Path]:
    index: dict[str, Path] = {}
    for directory in reversed(profile_search_paths()):  # user dir wins
        for p in sorted(directory.glob("*.y*ml")):
            index[p.stem] = p
    return index


def list_profiles() -> list[Profile]:
    """Every discoverable profile, built-ins plus the user directory."""
    out: list[Profile] = []
    for name, path in sorted(_builtin_index().items()):
        try:
            out.append(load_profile_file(path))
        except ProfileError as exc:  # a broken user profile must not hide the rest
            placeholder = Profile(name="generic")
            placeholder.notes = f"FAILED TO LOAD ({name}): {exc}"
            placeholder.source_path = str(path)
            out.append(placeholder)
    return out


def load_profile(name_or_path: str) -> Profile:
    """Load by built-in name, by file path, or raise with the known names."""
    if name_or_path in ("", "auto"):
        raise ProfileError("'auto' is resolved by fingerprinting, not by the loader")
    candidate = Path(name_or_path)
    if candidate.suffix.lower() in (".yaml", ".yml") or candidate.exists():
        return load_profile_file(candidate)
    index = _builtin_index()
    if name_or_path in index:
        return load_profile_file(index[name_or_path])
    known = ", ".join(sorted(index)) or "(none found)"
    raise ProfileError(f"unknown profile {name_or_path!r}; available: {known}")


def apply_overrides(profile: Profile, overrides: dict[str, Any]) -> Profile:
    """Apply dotted-key overrides from the CLI, e.g. ``segmentation.bg_tolerance=24``.

    Returns a new validated profile so a bad override is rejected the same way
    a bad YAML file is.
    """
    if not overrides:
        return profile
    data = profile.model_dump(mode="json")
    for key, value in overrides.items():
        parts = key.split(".")
        node: Any = data
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ProfileError(f"--set {key}: no such profile section {part!r}")
            node = node[part]
        leaf = parts[-1]
        if leaf not in node:
            raise ProfileError(f"--set {key}: no such profile key {leaf!r}")
        node[leaf] = value
    out = parse_profile(data, source=f"{profile.source_path} + overrides")
    return out


def parse_override(text: str) -> tuple[str, Any]:
    """Parse ``key.path=value``; the value is read as YAML so types come free."""
    if "=" not in text:
        raise ProfileError(f"--set needs key=value, got {text!r}")
    key, _, raw = text.partition("=")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ProfileError(f"--set {key}: cannot parse value {raw!r}: {exc}") from exc
    return key.strip(), value
