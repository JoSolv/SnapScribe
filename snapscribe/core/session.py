"""The ``Session``: every stage's output in one resumable, JSON-backed object."""

from __future__ import annotations

import datetime as dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from snapscribe.core.models import (
    ChromeInfo,
    Extraction,
    ImageRef,
    Message,
    Region,
    RegionKind,
    StageStatus,
)
from snapscribe.core.serde import from_dict, from_jsonable, to_jsonable

SESSION_FORMAT_VERSION = 1

STAGE_ORDER: tuple[str, ...] = (
    "ingest",
    "normalize",
    "layout",
    "extract",
    "verify",
    "assemble",
    "export",
)


@dataclass
class StageRecord:
    """Bookkeeping for one stage run, so the GUI can show per-stage state."""

    name: str
    status: StageStatus = StageStatus.PENDING
    started_at: dt.datetime | None = None
    finished_at: dt.datetime | None = None
    duration_s: float = 0.0
    error: str = ""
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class Session:
    """Mutable pipeline state.

    Stages read what earlier stages wrote and append their own output. Nothing
    is thrown away, which is what makes ``--from <stage>`` and the planned
    stage-by-stage GUI view possible.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    format_version: int = SESSION_FORMAT_VERSION
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now().astimezone())
    updated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now().astimezone())
    name: str = ""
    inputs: list[str] = field(default_factory=list)
    workdir: str = ""

    images: list[ImageRef] = field(default_factory=list)
    chrome: dict[str, ChromeInfo] = field(default_factory=dict)
    regions: dict[str, list[Region]] = field(default_factory=dict)
    extractions: dict[str, list[Extraction]] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)

    profile_name: str = ""
    profile_source: str = ""
    """builtin | file | override | auto."""
    profile_match: dict[str, Any] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, StageRecord] = field(default_factory=dict)
    path: str = ""

    # ---------------------------------------------------------------- lookups

    def image(self, image_id: str) -> ImageRef | None:
        return next((i for i in self.images if i.id == image_id), None)

    def regions_for(self, image_id: str, kind: RegionKind | None = None) -> list[Region]:
        out = self.regions.get(image_id, [])
        if kind is not None:
            out = [r for r in out if r.kind == kind]
        return out

    def all_regions(self, kind: RegionKind | None = None) -> list[Region]:
        out: list[Region] = []
        for img in self.images:
            out.extend(self.regions_for(img.id, kind))
        return out

    def region(self, region_id: str) -> Region | None:
        for regions in self.regions.values():
            for r in regions:
                if r.id == region_id:
                    return r
        return None

    def extraction(self, region_id: str, engine: str) -> Extraction | None:
        return next(
            (e for e in self.extractions.get(region_id, []) if e.engine == engine),
            None,
        )

    def add_extraction(self, extraction: Extraction) -> None:
        bucket = self.extractions.setdefault(extraction.region_id, [])
        for i, existing in enumerate(bucket):
            if existing.engine == extraction.engine:
                bucket[i] = extraction
                return
        bucket.append(extraction)

    def engines_used(self) -> list[str]:
        seen: list[str] = []
        for bucket in self.extractions.values():
            for e in bucket:
                if e.engine not in seen:
                    seen.append(e.engine)
        return seen

    # ----------------------------------------------------------- stage status

    def stage(self, name: str) -> StageRecord:
        return self.stages.setdefault(name, StageRecord(name=name))

    def is_done(self, name: str) -> bool:
        return self.stages.get(name, StageRecord(name=name)).status is StageStatus.DONE

    def invalidate_from(self, name: str) -> None:
        """Mark ``name`` and every later stage as pending (a rerun is coming)."""
        if name not in STAGE_ORDER:
            return
        start = STAGE_ORDER.index(name)
        for later in STAGE_ORDER[start:]:
            if later in self.stages:
                self.stages[later].status = StageStatus.PENDING

    # --------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        session = from_dict(cls, data)
        # ``from_dict`` handles nested dataclasses generically; the two dict
        # fields with list values need their element type spelled out.
        session.regions = from_jsonable(dict[str, list[Region]], data.get("regions", {}))
        session.extractions = from_jsonable(
            dict[str, list[Extraction]], data.get("extractions", {})
        )
        session.messages = from_jsonable(list[Message], data.get("messages", []))
        return session

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or self.path)
        if not target:
            raise ValueError("no session path given")
        if target.is_dir():
            target = target / "session.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = dt.datetime.now().astimezone()
        self.path = str(target)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> Session:
        p = Path(path)
        if p.is_dir():
            p = p / "session.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        version = data.get("format_version", 0)
        if version > SESSION_FORMAT_VERSION:
            raise ValueError(
                f"session {p} was written by a newer SnapScribe "
                f"(format {version} > {SESSION_FORMAT_VERSION})"
            )
        session = cls.from_dict(data)
        session.path = str(p)
        return session

    # --------------------------------------------------------------- summary

    def summary(self) -> dict[str, Any]:
        bubbles = self.all_regions(RegionKind.BUBBLE)
        return {
            "id": self.id,
            "images": len(self.images),
            "bubbles": len(bubbles),
            "messages": len(self.messages),
            "needs_review": sum(1 for m in self.messages if m.needs_review),
            "profile": self.profile_name,
            "stages": {k: v.status.value for k, v in self.stages.items()},
        }
