"""Assemble stage: messages per screenshot, stitched into one conversation."""

from __future__ import annotations

import datetime as dt

from snapscribe.assemble.messages import anchor_date, build_messages, resolve_timestamps
from snapscribe.assemble.stitch import DEFAULT_THRESHOLD, stitch_messages
from snapscribe.core.models import Message, Side
from snapscribe.core.pipeline import PipelineContext
from snapscribe.core.session import Session
from snapscribe.verify.rules import VerifyConfig, check_messages


class AssembleStage:
    name = "assemble"
    requires = ("extract",)

    def run(self, session: Session, ctx: PipelineContext) -> None:
        profile = ctx.require_profile()
        verify_settings = session.settings.get("verify", {})
        primary = verify_settings.get("primary") or "ocr"
        secondary = verify_settings.get("secondary") or ""

        anchor = ctx.option("anchor_date")
        anchor = dt.date.fromisoformat(anchor) if isinstance(anchor, str) and anchor else None

        all_messages, notes = build_messages(
            session,
            profile,
            primary_engine=primary,
            secondary_engine=secondary,
            prefer=ctx.option("prefer", "primary"),
        )

        per_image: list[tuple[str, list[Message]]] = []
        for ref in session.images:
            per_image.append((ref.name, [m for m in all_messages if m.source_image == ref.name]))

        if ctx.option("stitch", True):
            merged, report = stitch_messages(
                per_image, threshold=ctx.option("stitch_threshold", DEFAULT_THRESHOLD)
            )
            notes.extend(report.notes)
            overlaps = report.overlaps
            removed = report.removed
        else:
            merged = all_messages
            for i, m in enumerate(merged):
                m.index = i
            overlaps, removed = [], 0

        notes.extend(resolve_timestamps(merged, profile, anchor or anchor_date(session)))

        if ctx.option("drop_separators", True):
            merged = [m for m in merged if m.meta.get("kind") != "date_separator"]
            for i, m in enumerate(merged):
                m.index = i

        cfg: VerifyConfig = ctx.option("verify_config") or VerifyConfig()
        rules = check_messages(merged, cfg)

        if profile.senders.group_chat or session.settings.get("group_chat"):
            unnamed = sum(
                1 for m in merged if m.side is Side.LEFT and m.sender in ("", profile.senders.other)
            )
            if unnamed:
                notes.append(
                    f"{unnamed} incoming message(s) have no sender name. A name is only "
                    "trusted once it appears above two or more bubbles; for someone who "
                    "speaks just once, list them with "
                    "--set senders.known_senders='[\"Name\"]'"
                )

        session.messages = merged
        session.settings["stitch"] = {"overlaps": overlaps, "removed": removed}
        record = session.stage(self.name)
        record.counts = {
            "messages": len(merged),
            "duplicates_removed": removed,
            "needs_review": rules.flagged,
            "with_timestamp": sum(1 for m in merged if m.timestamp is not None),
        }
        record.notes = notes + [f"{k}: {v}" for k, v in sorted(rules.reasons.items())]
        ctx.report(self.name, 1.0, f"{len(merged)} messages")
