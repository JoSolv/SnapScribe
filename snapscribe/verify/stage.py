"""Verify stage: diff the engines per region and store the verdicts.

Runs *before* assembly so that assembly can pick the better of the two texts
per region; the rule checks that need a message list run again at the end of
assembly.
"""

from __future__ import annotations

from snapscribe.core.pipeline import PipelineContext
from snapscribe.core.session import Session
from snapscribe.verify.compare import compare_texts
from snapscribe.verify.rules import VerifyConfig


class VerifyStage:
    name = "verify"
    requires = ("extract",)

    def run(self, session: Session, ctx: PipelineContext) -> None:
        cfg: VerifyConfig = ctx.option("verify_config") or VerifyConfig()
        engines = session.engines_used()
        primary = "ocr" if "ocr" in engines else (engines[0] if engines else "")
        secondary = next((e for e in engines if e != primary), "")

        agree = disagree = single = 0
        for region_id, bucket in session.extractions.items():
            texts = {e.engine: e for e in bucket}
            a = texts.get(primary)
            b = texts.get(secondary) if secondary else None
            if a is None:
                continue
            if b is None:
                a.meta["agreement"] = None
                single += 1
                continue
            comparison = compare_texts(a.text, b.text)
            verdict = {
                "agreement": round(comparison.similarity, 4),
                "token_agreement": round(comparison.token_similarity, 4),
                "only_in_" + primary: comparison.only_in_a[:12],
                "only_in_" + secondary: comparison.only_in_b[:12],
            }
            a.meta.update(verdict)
            b.meta.update(verdict)
            if comparison.similarity >= cfg.agreement_threshold:
                agree += 1
            else:
                disagree += 1
            ctx.check_cancelled()
            del region_id

        record = session.stage(self.name)
        record.counts = {"agree": agree, "disagree": disagree, "single_engine": single}
        record.notes = [
            f"primary engine: {primary or 'none'}",
            f"secondary engine: {secondary or 'none (single-engine run)'}",
            f"agreement threshold: {cfg.agreement_threshold}",
        ]
        session.settings["verify"] = {
            "primary": primary,
            "secondary": secondary,
            "agreement_threshold": cfg.agreement_threshold,
        }
