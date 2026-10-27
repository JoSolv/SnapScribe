"""Stage protocol and runner.

Each stage is a small object with a name, declared dependencies and a ``run``
that mutates the session in place. The runner handles ordering, skipping
already-done stages, timing, error capture and autosaving — so a crash in
``extract`` still leaves a session with usable ``layout`` output on disk.
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from snapscribe.core.models import StageStatus
from snapscribe.core.session import STAGE_ORDER, Session

if TYPE_CHECKING:  # pragma: no cover - typing only
    from snapscribe.profiles.schema import Profile

ProgressFn = Callable[[str, float, str], None]
"""``(stage, fraction_0_to_1, message)``; fraction < 0 means indeterminate."""


class StageError(RuntimeError):
    """A stage failed in a way that should stop the run."""


class Cancelled(RuntimeError):
    """The caller asked to stop (GUI cancel button)."""


@dataclass
class PipelineContext:
    """Everything a stage needs besides the session itself."""

    profile: Profile | None = None
    profile_request: str = "auto"
    debug_dir: Path | None = None
    options: dict[str, Any] = field(default_factory=dict)
    progress: ProgressFn | None = None
    should_cancel: Callable[[], bool] | None = None
    autosave: bool = True

    def report(self, stage: str, fraction: float, message: str = "") -> None:
        if self.progress is not None:
            self.progress(stage, fraction, message)

    def check_cancelled(self) -> None:
        if self.should_cancel is not None and self.should_cancel():
            raise Cancelled("cancelled by caller")

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def require_profile(self) -> Profile:
        if self.profile is None:
            raise StageError("this stage needs a profile; run the normalize stage first")
        return self.profile


@runtime_checkable
class Stage(Protocol):
    """Pipeline stage interface."""

    name: str
    requires: tuple[str, ...]

    def run(self, session: Session, ctx: PipelineContext) -> None: ...


def default_stages() -> list[Stage]:
    """Build the standard stage list.

    Imported lazily so that `--help` and the ingest-only path do not pay for
    OpenCV/OCR imports.
    """
    from snapscribe.assemble.stage import AssembleStage
    from snapscribe.export.stage import ExportStage
    from snapscribe.extract.stage import ExtractStage
    from snapscribe.ingest.stage import IngestStage
    from snapscribe.layout.stage import LayoutStage
    from snapscribe.normalize.stage import NormalizeStage
    from snapscribe.verify.stage import VerifyStage

    return [
        IngestStage(),
        NormalizeStage(),
        LayoutStage(),
        ExtractStage(),
        VerifyStage(),
        AssembleStage(),
        ExportStage(),
    ]


def _select(stages: Sequence[Stage], from_stage: str | None, to_stage: str | None) -> list[Stage]:
    names = [s.name for s in stages]
    start = 0
    end = len(stages)
    if from_stage:
        if from_stage not in names:
            raise StageError(f"unknown stage {from_stage!r}; known: {', '.join(names)}")
        start = names.index(from_stage)
    if to_stage:
        if to_stage not in names:
            raise StageError(f"unknown stage {to_stage!r}; known: {', '.join(names)}")
        end = names.index(to_stage) + 1
    if start >= end:
        raise StageError(f"stage range {from_stage}..{to_stage} is empty")
    return list(stages[start:end])


def run_pipeline(
    session: Session,
    ctx: PipelineContext,
    stages: Iterable[Stage] | None = None,
    from_stage: str | None = None,
    to_stage: str | None = None,
    force: bool = False,
    skip_done: bool = False,
) -> Session:
    """Run the selected stages in order, recording status on the session.

    ``from_stage`` marks that stage and everything after it as pending, so a
    rerun of one stage does not leave stale downstream output claiming to be
    current. ``skip_done`` resumes an interrupted run by stepping over stages
    already marked done.
    """
    all_stages = list(stages) if stages is not None else default_stages()
    selected = _select(all_stages, from_stage, to_stage)

    if from_stage and not force:
        session.invalidate_from(from_stage)

    done_names = {s.name for s in all_stages if session.is_done(s.name)}

    for stage in selected:
        ctx.check_cancelled()
        record = session.stage(stage.name)

        if skip_done and not force and record.status is StageStatus.DONE:
            done_names.add(stage.name)
            ctx.report(stage.name, 1.0, f"{stage.name}: already done, skipping")
            continue

        missing = [r for r in getattr(stage, "requires", ()) if r not in done_names]
        if missing:
            raise StageError(
                f"stage {stage.name!r} needs {', '.join(missing)} to have run first "
                f"(use --from {missing[0]} or run the full pipeline)"
            )

        record.status = StageStatus.RUNNING
        record.started_at = dt.datetime.now().astimezone()
        record.error = ""
        started = time.perf_counter()
        ctx.report(stage.name, -1.0, f"{stage.name}: starting")
        try:
            stage.run(session, ctx)
        except Cancelled:
            record.status = StageStatus.PENDING
            record.error = "cancelled"
            _finish(record, started)
            _autosave(session, ctx)
            raise
        except Exception as exc:
            record.status = StageStatus.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            _finish(record, started)
            _autosave(session, ctx)
            raise
        record.status = StageStatus.DONE
        _finish(record, started)
        done_names.add(stage.name)
        ctx.report(stage.name, 1.0, f"{stage.name}: done")
        _autosave(session, ctx)

    return session


def _finish(record: Any, started: float) -> None:
    record.finished_at = dt.datetime.now().astimezone()
    record.duration_s = round(time.perf_counter() - started, 3)


def _autosave(session: Session, ctx: PipelineContext) -> None:
    if ctx.autosave and session.path:
        session.save()


__all__ = [
    "STAGE_ORDER",
    "Cancelled",
    "PipelineContext",
    "ProgressFn",
    "Stage",
    "StageError",
    "default_stages",
    "run_pipeline",
]
