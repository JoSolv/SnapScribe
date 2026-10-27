"""Export stage."""

from __future__ import annotations

from pathlib import Path

from snapscribe.core.pipeline import PipelineContext, StageError
from snapscribe.core.session import Session
from snapscribe.export.csv_writer import write_csv


class ExportStage:
    name = "export"
    requires = ("assemble",)

    def run(self, session: Session, ctx: PipelineContext) -> None:
        output = ctx.option("output")
        if not output:
            raise StageError("export needs an output path (-o/--output)")
        if not session.messages:
            raise StageError("nothing to export: assemble produced no messages")
        path = write_csv(
            Path(output),
            session.messages,
            include_debug=ctx.option("include_debug", True),
            delimiter=ctx.option("delimiter", ","),
        )
        record = session.stage(self.name)
        record.counts = {"rows": len(session.messages)}
        record.notes = [f"wrote {path}"]
        session.settings["output"] = str(path)
        ctx.report(self.name, 1.0, str(path))
