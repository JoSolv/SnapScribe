"""Profile loading, validation, overrides and fingerprinting."""

from __future__ import annotations

import numpy as np
import pytest

from snapscribe.imaging import hex_to_bgr
from snapscribe.profiles import (
    Profile,
    ProfileError,
    apply_overrides,
    detect_profile,
    list_profiles,
    load_profile,
    parse_override,
    parse_profile,
    score_profile,
)


def test_builtin_profiles_load():
    names = {p.name for p in list_profiles()}
    assert {"whatsapp_light", "whatsapp_dark", "generic"} <= names


def test_unknown_profile_lists_the_known_ones():
    with pytest.raises(ProfileError, match="whatsapp_light"):
        load_profile("nope")


def test_malformed_profile_reports_the_field():
    with pytest.raises(ProfileError) as exc:
        parse_profile({"name": "x", "colors": {"background": "not-a-colour"}}, "test.yaml")
    assert "colors.background" in str(exc.value)


def test_unknown_key_is_rejected():
    with pytest.raises(ProfileError, match="typo"):
        parse_profile({"name": "x", "typo": 1}, "test.yaml")


def test_margins_must_leave_a_chat_area():
    with pytest.raises(ProfileError, match="chat area"):
        parse_profile(
            {"name": "x", "layout": {"header_fraction": 0.7, "footer_fraction": 0.5}}, "t.yaml"
        )


def test_overrides_apply_and_validate():
    profile = load_profile("whatsapp_light")
    key, value = parse_override("segmentation.bg_tolerance=27")
    assert apply_overrides(profile, {key: value}).segmentation.bg_tolerance == 27
    with pytest.raises(ProfileError, match="no such profile key"):
        apply_overrides(profile, {"segmentation.nonsense": 1})
    with pytest.raises(ProfileError, match="no such profile section"):
        apply_overrides(profile, {"nope.x": 1})


def _chat_area(background: str, incoming: str, outgoing: str) -> np.ndarray:
    image = np.full((400, 300, 3), hex_to_bgr(background), np.uint8)
    image[40:120, 10:200] = hex_to_bgr(incoming)
    image[160:240, 100:290] = hex_to_bgr(outgoing)
    return image


def test_fingerprint_picks_the_matching_theme():
    light = _chat_area("#EFEAE2", "#FFFFFF", "#D9FDD3")
    dark = _chat_area("#0B141A", "#202C33", "#005C4B")
    profiles = list_profiles()

    chosen, match, _ = detect_profile(light, profiles)
    assert chosen.name == "whatsapp_light"
    assert match.score > 0.8

    chosen, _, _ = detect_profile(dark, profiles)
    assert chosen.name == "whatsapp_dark"


def test_fingerprint_falls_back_to_generic_for_an_unknown_app():
    alien = _chat_area("#3A1F5D", "#7A4FB6", "#FF8800")
    chosen, match, ranking = detect_profile(alien, list_profiles())
    assert chosen.name == "generic"
    assert "reason" in match.details
    assert ranking


def test_footer_text_lifts_the_score():
    light = _chat_area("#EFEAE2", "#FFFFFF", "#D9FDD3")
    profile = load_profile("whatsapp_light")
    without = score_profile(light, profile)
    with_text = score_profile(light, profile, footer_text="Type a message")
    assert with_text.score >= without.score
    assert with_text.signals["footer_text"] == 1.0


def test_profile_yaml_round_trips():
    profile = load_profile("whatsapp_dark")
    import yaml

    reparsed = parse_profile(yaml.safe_load(profile.dump_yaml()), "round-trip")
    assert isinstance(reparsed, Profile)
    assert reparsed.colors.outgoing_bubble == profile.colors.outgoing_bubble
