"""The corpus manifest: which conversations get rendered in which themes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from snapscribe.corpus.render import RenderResult, RenderSpec, build_corpus

SOURCE_DIR = Path("corpus/sources")
OUTPUT_DIR = Path("corpus/generated")


@dataclass(frozen=True)
class CorpusSet:
    """One rendered set: a conversation in one theme on one device size."""

    name: str
    source: str
    profile: str
    owner: str = "Me"
    group_chat: bool | None = None
    device: tuple[int, int] = (1080, 2340)
    avatar_px: int = 0
    jitter: float = 0.35
    seed: int = 7
    notes: str = ""


DEFAULT_SETS: tuple[CorpusSet, ...] = (
    CorpusSet(
        name="whatsapp_light",
        source="coffee_1to1.txt",
        profile="whatsapp_light",
        notes="1:1 chat, light theme, a document attachment, spans two days",
    ),
    CorpusSet(
        name="whatsapp_dark",
        source="coffee_1to1.txt",
        profile="whatsapp_dark",
        notes="same conversation, dark theme: the hard case for background separation",
    ),
    CorpusSet(
        name="group_light",
        source="trip_group.txt",
        profile="whatsapp_light",
        group_chat=True,
        avatar_px=64,
        notes="4-person group, sender names in bubbles, avatars, a system message",
    ),
    CorpusSet(
        name="group_dark",
        source="trip_group.txt",
        profile="whatsapp_dark",
        group_chat=True,
        avatar_px=64,
        seed=11,
        notes="group chat in dark theme",
    ),
    CorpusSet(
        name="media_light",
        source="media_emoji.txt",
        profile="whatsapp_light",
        seed=3,
        notes="colour emoji, image/video/audio/sticker bubbles, a deleted message",
    ),
    CorpusSet(
        name="media_dark",
        source="media_emoji.txt",
        profile="whatsapp_dark",
        seed=5,
        notes="emoji and media in dark theme",
    ),
    CorpusSet(
        name="tablet_light",
        source="coffee_1to1.txt",
        profile="whatsapp_light",
        device=(1536, 2048),
        seed=13,
        notes="different aspect ratio, to catch hard-coded phone geometry",
    ),
)


def build_set(
    spec: CorpusSet, source_dir: Path = SOURCE_DIR, output_dir: Path = OUTPUT_DIR
) -> RenderResult:
    render_spec = RenderSpec(
        width=spec.device[0],
        height=spec.device[1],
        avatar_px=spec.avatar_px,
        jitter=spec.jitter,
    )
    return build_corpus(
        export_path=source_dir / spec.source,
        out_dir=output_dir / spec.name,
        profile_name=spec.profile,
        owner=spec.owner,
        group_chat=spec.group_chat,
        spec=render_spec,
        seed=spec.seed,
        prefix=Path(spec.source).stem,
    )


def build_all(
    sets: tuple[CorpusSet, ...] = DEFAULT_SETS,
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, RenderResult]:
    return {s.name: build_set(s, source_dir, output_dir) for s in sets}
