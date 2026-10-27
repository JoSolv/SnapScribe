"""Command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from snapscribe import __version__
from snapscribe.core.pipeline import Cancelled, PipelineContext, StageError, run_pipeline
from snapscribe.core.session import STAGE_ORDER, Session
from snapscribe.profiles.loader import ProfileError

app = typer.Typer(
    name="snapscribe",
    help="Transcribe chat conversations from smartphone screenshots.",
    no_args_is_help=True,
    add_completion=False,
)
profiles_app = typer.Typer(name="profile", help="Inspect and test profiles.", no_args_is_help=True)
corpus_app = typer.Typer(
    name="corpus", help="Build and score the ground-truth corpus.", no_args_is_help=True
)
session_app = typer.Typer(name="session", help="Inspect saved sessions.", no_args_is_help=True)
app.add_typer(profiles_app)
app.add_typer(corpus_app)
app.add_typer(session_app)

console = Console()
err_console = Console(stderr=True)


def _fail(message: str) -> None:
    err_console.print(f"[bold red]error:[/] {message}")
    raise typer.Exit(code=1)


def _progress_printer(verbose: bool):
    if not verbose:
        return None

    def report(stage: str, fraction: float, message: str) -> None:
        bar = "" if fraction < 0 else f" {fraction * 100:3.0f}%"
        console.print(f"[dim]{stage}{bar}[/] {message}")

    return report


def _parse_overrides(values: list[str] | None) -> dict[str, object]:
    from snapscribe.profiles import parse_override

    out: dict[str, object] = {}
    for item in values or []:
        key, value = parse_override(item)
        out[key] = value
    return out


def _build_context(
    profile: str,
    engines: list[str],
    overlay: Path | None,
    options: dict[str, object],
    verbose: bool,
) -> PipelineContext:
    return PipelineContext(
        profile_request=profile,
        debug_dir=overlay,
        options={"engines": engines, **options},
        progress=_progress_printer(verbose),
    )


def _engine_list(engine: str, vision: bool) -> list[str]:
    names = [e.strip() for e in engine.split(",") if e.strip()]
    if vision and "vision" not in names:
        names.append("vision")
    return names or ["ocr"]


@app.callback()
def main_callback(
    version: Annotated[bool, typer.Option("--version", help="Print the version and exit.")] = False,
) -> None:
    if version:
        console.print(f"snapscribe {__version__}")
        raise typer.Exit()


@app.command()
def ingest(
    inputs: Annotated[list[Path], typer.Argument(help="Folders, zips or image files.")],
    session_path: Annotated[
        Path, typer.Option("--session", "-s", help="Where to save the session.")
    ] = Path("sessions/session.json"),
    order: Annotated[str, typer.Option(help="auto | name | exif | mtime")] = "auto",
    recursive: Annotated[bool, typer.Option(help="Descend into subfolders.")] = True,
) -> None:
    """Import images, put them in order and save a session."""
    session = Session(workdir=str(Path(session_path).parent))
    ctx = _build_context(
        "auto", ["none"], None, {"inputs": inputs, "order": order, "recursive": recursive}, False
    )
    try:
        run_pipeline(session, ctx, to_stage="ingest")
    except StageError as exc:
        _fail(str(exc))

    table = Table(title=f"{len(session.images)} image(s)", header_style="bold")
    for column in ("#", "file", "size", "taken at", "order from", "kind"):
        table.add_column(column)
    for ref in session.images:
        table.add_row(
            str(ref.index),
            ref.name,
            f"{ref.width}x{ref.height}",
            ref.taken_at.strftime("%Y-%m-%d %H:%M:%S") if ref.taken_at else "-",
            ref.order_source,
            ref.source_kind.value,
        )
    console.print(table)
    for note in session.stage("ingest").notes:
        console.print(f"[yellow]note:[/] {note}")
    saved = session.save(session_path)
    console.print(f"session saved to [bold]{saved}[/]")


@app.command()
def run(
    inputs: Annotated[
        list[Path] | None, typer.Argument(help="Folders, zips or image files.")
    ] = None,
    output: Annotated[Path, typer.Option("--output", "-o", help="CSV to write.")] = Path(
        "out/transcript.csv"
    ),
    profile: Annotated[
        str, typer.Option("--profile", "-p", help="Profile name, YAML path, or 'auto'.")
    ] = "auto",
    engine: Annotated[
        str, typer.Option("--engine", "-e", help="Comma-separated: ocr, vision, none.")
    ] = "ocr",
    vision: Annotated[bool, typer.Option("--vision", help="Also run the vision model.")] = False,
    vision_model: Annotated[str, typer.Option(help="Ollama model for the vision path.")] = "",
    vision_host: Annotated[str, typer.Option(help="Ollama base URL.")] = "",
    overlay: Annotated[
        Path | None, typer.Option("--overlay", help="Write debug overlays here.")
    ] = None,
    session_path: Annotated[
        Path | None, typer.Option("--session", "-s", help="Session file to write or resume.")
    ] = None,
    from_stage: Annotated[
        str | None, typer.Option("--from", help=f"Rerun from a stage: {', '.join(STAGE_ORDER)}")
    ] = None,
    to_stage: Annotated[str | None, typer.Option("--to", help="Stop after this stage.")] = None,
    anchor_date: Annotated[
        str, typer.Option(help="Date that 'Today' refers to (YYYY-MM-DD).")
    ] = "",
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", help="Override a profile value, e.g. segmentation.bg_tolerance=24."),
    ] = None,
    group_chat: Annotated[
        bool,
        typer.Option("--group/--no-group", help="Treat as a group chat (sender names in bubbles)."),
    ] = False,
    no_stitch: Annotated[
        bool,
        typer.Option("--no-stitch", help="Keep duplicate messages from overlapping screenshots."),
    ] = False,
    prefer: Annotated[
        str, typer.Option(help="Which engine's text wins: primary | vision | longest.")
    ] = "primary",
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
) -> None:
    """Run the full pipeline: ingest -> ... -> export."""
    if session_path and Path(session_path).exists() and not inputs:
        session = Session.load(session_path)
    else:
        if not inputs:
            _fail("give at least one input path, or --session pointing at an existing session")
        session = Session(workdir=str(Path(session_path).parent) if session_path else "")
    if session_path:
        session.path = str(session_path)

    try:
        overrides = _parse_overrides(set_)
    except ProfileError as exc:
        _fail(str(exc))

    options: dict[str, object] = {
        "inputs": inputs or session.inputs,
        "output": output,
        "profile_overrides": overrides,
        "anchor_date": anchor_date,
        "stitch": not no_stitch,
        "prefer": prefer,
        "vision_model": vision_model,
        "vision_host": vision_host,
    }
    session.settings["group_chat"] = group_chat
    ctx = _build_context(profile, _engine_list(engine, vision), overlay, options, not quiet)

    try:
        run_pipeline(
            session, ctx, from_stage=from_stage, to_stage=to_stage, skip_done=bool(from_stage)
        )
    except Cancelled:
        _fail("cancelled")
    except StageError as exc:
        _fail(str(exc))
    except ProfileError as exc:
        _fail(str(exc))

    _print_summary(session)
    if session.path:
        session.save()
        console.print(f"session saved to [bold]{session.path}[/]")


@app.command("layout")
def layout_cmd(
    inputs: Annotated[list[Path], typer.Argument(help="Folders, zips or image files.")],
    overlay: Annotated[
        Path, typer.Option("--overlay", help="Where to write debug overlays.")
    ] = Path("debug"),
    profile: Annotated[str, typer.Option("--profile", "-p")] = "auto",
    set_: Annotated[list[str] | None, typer.Option("--set")] = None,
    scale: Annotated[float, typer.Option(help="Scale the overlay images.")] = 1.0,
) -> None:
    """Run as far as layout and write the debug overlays. Use this to tune a profile."""
    session = Session()
    try:
        overrides = _parse_overrides(set_)
    except ProfileError as exc:
        _fail(str(exc))
    ctx = _build_context(
        profile,
        ["none"],
        overlay,
        {"inputs": inputs, "profile_overrides": overrides, "overlay_scale": scale},
        True,
    )
    try:
        run_pipeline(session, ctx, to_stage="layout")
    except (StageError, ProfileError) as exc:
        _fail(str(exc))
    _print_summary(session)
    console.print(f"overlays written to [bold]{overlay}[/]")
    from snapscribe.layout.overlay import legend_lines

    for line in legend_lines():
        console.print(f"  [dim]{line}[/]")


def _print_summary(session: Session) -> None:
    table = Table(title="pipeline", header_style="bold")
    for column in ("stage", "status", "seconds", "counts"):
        table.add_column(column)
    for name in STAGE_ORDER:
        record = session.stages.get(name)
        if record is None:
            continue
        counts = ", ".join(f"{k}={v}" for k, v in record.counts.items())
        colour = {"done": "green", "failed": "red"}.get(record.status.value, "yellow")
        table.add_row(
            name, f"[{colour}]{record.status.value}[/]", f"{record.duration_s:.2f}", counts
        )
    console.print(table)
    if session.profile_name:
        console.print(f"profile: [bold]{session.profile_name}[/] ({session.profile_source})")
    notes = [
        (name, note) for name in STAGE_ORDER for note in session.stages.get(name, _empty()).notes
    ]
    for name, note in notes:
        console.print(f"[yellow]{name}:[/] {note}")
    flagged = [m for m in session.messages if m.needs_review]
    if session.messages:
        console.print(
            f"[bold]{len(session.messages)}[/] messages, [bold]{len(flagged)}[/] need review"
        )


def _empty():
    from snapscribe.core.session import StageRecord

    return StageRecord(name="")


@app.command()
def review(
    session_path: Annotated[Path, typer.Argument(help="Session file to inspect.")],
    limit: Annotated[int, typer.Option(help="Maximum rows to show.")] = 20,
    all_rows: Annotated[
        bool, typer.Option("--all", help="Show every message, not just flagged ones.")
    ] = False,
) -> None:
    """Show the review queue from a saved session."""
    session = Session.load(session_path)
    rows = session.messages if all_rows else [m for m in session.messages if m.needs_review]
    if not rows:
        console.print("[green]nothing flagged for review[/]")
        return
    table = Table(header_style="bold")
    for column in ("#", "sender", "time", "body", "conf", "why"):
        table.add_column(column, overflow="fold")
    for message in rows[:limit]:
        table.add_row(
            str(message.index),
            message.sender,
            message.timestamp.strftime("%Y-%m-%d %H:%M") if message.timestamp else "-",
            message.body[:70],
            f"{message.confidence:.2f}",
            "; ".join(message.review_reasons),
        )
    console.print(table)
    if len(rows) > limit:
        console.print(f"[dim]... and {len(rows) - limit} more[/]")


@session_app.command("show")
def session_show(session_path: Annotated[Path, typer.Argument()]) -> None:
    """Print a session summary."""
    session = Session.load(session_path)
    console.print_json(json.dumps(session.summary(), indent=2))
    _print_summary(session)


@profiles_app.command("list")
def profile_list() -> None:
    """List the available profiles."""
    from snapscribe.profiles import list_profiles

    table = Table(header_style="bold")
    for column in ("name", "app", "theme", "source", "notes"):
        table.add_column(column, overflow="fold")
    for profile in list_profiles():
        table.add_row(
            profile.name,
            profile.app or "-",
            profile.theme,
            Path(profile.source_path).name if profile.source_path else "-",
            (profile.notes or "").strip().replace("\n", " ")[:70],
        )
    console.print(table)


@profiles_app.command("show")
def profile_show(name: Annotated[str, typer.Argument()]) -> None:
    """Print a profile as YAML."""
    from snapscribe.profiles import load_profile

    try:
        console.print(load_profile(name).dump_yaml())
    except ProfileError as exc:
        _fail(str(exc))


@profiles_app.command("detect")
def profile_detect(
    inputs: Annotated[list[Path], typer.Argument(help="Screenshots to fingerprint.")],
    no_ocr: Annotated[
        bool, typer.Option("--no-ocr", help="Skip reading the composer text.")
    ] = False,
) -> None:
    """Report which profile matches these screenshots, and why."""
    session = Session()
    ctx = _build_context(
        "auto", ["none"], None, {"inputs": inputs, "fingerprint_ocr": not no_ocr}, False
    )
    try:
        run_pipeline(session, ctx, to_stage="normalize")
    except (StageError, ProfileError) as exc:
        _fail(str(exc))
    match = session.profile_match
    console.print(f"chosen: [bold]{session.profile_name}[/]")
    if match.get("footer_text"):
        console.print(f"composer text read as: [italic]{match['footer_text'][:60]}[/]")
    table = Table(header_style="bold")
    for column in ("profile", "score", "signals"):
        table.add_column(column, overflow="fold")
    for entry in match.get("ranking", []):
        signals = ", ".join(f"{k}={v:.2f}" for k, v in entry["signals"].items())
        table.add_row(entry["name"], f"{entry['score']:.3f}", signals)
    console.print(table)
    for image_id, chrome in session.chrome.items():
        console.print(
            f"[dim]{image_id}: header {chrome.header_height}px, footer {chrome.footer_height}px "
            f"({chrome.method}, confidence {chrome.confidence:.2f})[/]"
        )


@corpus_app.command("build")
def corpus_build(
    only: Annotated[str, typer.Option(help="Comma-separated set names; default is all.")] = "",
    source_dir: Annotated[Path, typer.Option()] = Path("corpus/sources"),
    output_dir: Annotated[Path, typer.Option()] = Path("corpus/generated"),
) -> None:
    """Render the synthetic corpus from the export files in corpus/sources."""
    from snapscribe.corpus.build import DEFAULT_SETS, build_set

    wanted = {s.strip() for s in only.split(",") if s.strip()}
    sets = [s for s in DEFAULT_SETS if not wanted or s.name in wanted]
    if not sets:
        _fail(f"no corpus set matches {only!r}; known: {', '.join(s.name for s in DEFAULT_SETS)}")
    table = Table(header_style="bold")
    for column in ("set", "images", "messages", "profile", "notes"):
        table.add_column(column, overflow="fold")
    for spec in sets:
        result = build_set(spec, source_dir, output_dir)
        meta = json.loads(result.meta_json.read_text()) if result.meta_json else {}
        table.add_row(
            spec.name,
            str(len(result.images)),
            str(meta.get("message_count", "?")),
            spec.profile,
            spec.notes,
        )
        for warning in result.warnings:
            err_console.print(f"[yellow]{spec.name}: {warning}[/]")
    console.print(table)
    console.print(f"corpus written to [bold]{output_dir}[/]")


@corpus_app.command("score")
def corpus_score(
    predicted: Annotated[Path, typer.Argument(help="CSV produced by `snapscribe run`.")],
    ground_truth: Annotated[
        Path, typer.Argument(help="ground_truth.csv or a WhatsApp export .txt")
    ],
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write the report as JSON.")
    ] = None,
    include_system: Annotated[bool, typer.Option(help="Score system messages too.")] = False,
    worst: Annotated[int, typer.Option(help="How many bad rows to print.")] = 5,
) -> None:
    """Score a transcript against ground truth."""
    from snapscribe.corpus.score import score_csv, write_report

    try:
        report = score_csv(predicted, ground_truth, include_system=include_system, worst_n=worst)
    except FileNotFoundError as exc:
        _fail(str(exc))
    console.print(report.format_text())
    if json_out:
        write_report(report, json_out)
        console.print(f"report written to [bold]{json_out}[/]")


@corpus_app.command("list")
def corpus_list(output_dir: Annotated[Path, typer.Option()] = Path("corpus/generated")) -> None:
    """List the generated corpus sets."""
    table = Table(header_style="bold")
    for column in ("set", "images", "messages", "profile", "notes"):
        table.add_column(column, overflow="fold")
    for meta_path in sorted(Path(output_dir).glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        table.add_row(
            meta_path.parent.name,
            str(len(meta.get("images", []))),
            str(meta.get("message_count", "?")),
            meta.get("profile", "-"),
            f"{meta.get('theme', '')} {'group' if meta.get('group_chat') else ''}".strip(),
        )
    console.print(table)


@app.command()
def bench(
    corpus_set: Annotated[Path, typer.Argument(help="A generated corpus directory.")],
    engines: Annotated[str, typer.Option(help="Comma-separated engines to compare.")] = "ocr",
    models: Annotated[str, typer.Option(help="Comma-separated vision models to compare.")] = "",
    profile: Annotated[str, typer.Option("--profile", "-p")] = "auto",
    out_dir: Annotated[Path, typer.Option()] = Path("out/bench"),
) -> None:
    """Benchmark engines (and vision models) against a corpus set's ground truth."""
    from snapscribe.corpus.score import score_csv

    ground_truth = Path(corpus_set) / "ground_truth.csv"
    if not ground_truth.exists():
        _fail(f"{ground_truth} not found; build the corpus first")

    runs: list[tuple[str, list[str], str]] = []
    for name in [e.strip() for e in engines.split(",") if e.strip()]:
        if name == "vision" and models:
            runs.extend((f"vision:{m.strip()}", ["vision"], m.strip()) for m in models.split(","))
        else:
            runs.append((name, [name], ""))

    table = Table(header_style="bold")
    for column in ("engine", "messages", "recall", "CER", "exact", "sender", "time s"):
        table.add_column(column)
    for label, engine_names, model in runs:
        session = Session()
        output = Path(out_dir) / f"{Path(corpus_set).name}_{label.replace(':', '_')}.csv"
        ctx = _build_context(
            profile,
            engine_names,
            None,
            {"inputs": [corpus_set], "output": output, "vision_model": model},
            False,
        )
        import time as _time

        started = _time.perf_counter()
        try:
            run_pipeline(session, ctx)
        except (StageError, ProfileError) as exc:
            err_console.print(f"[red]{label}: {exc}[/]")
            continue
        elapsed = _time.perf_counter() - started
        report = score_csv(output, ground_truth)
        table.add_row(
            label,
            f"{report.pred_count}/{report.gt_count}",
            f"{report.recall:.1%}",
            f"{report.cer:.1%}",
            f"{report.exact_body:.1%}",
            f"{report.sender_accuracy:.1%}",
            f"{elapsed:.1f}",
        )
    console.print(table)


@app.command()
def engines() -> None:
    """Show which extraction engines are usable here."""
    from snapscribe.extract import available_engines, get_extractor

    table = Table(header_style="bold")
    for column in ("engine", "available", "details"):
        table.add_column(column)
    for name in available_engines():
        engine = get_extractor(name)
        ok = engine.available()
        table.add_row(name, "[green]yes[/]" if ok else "[red]no[/]", engine.describe())
    console.print(table)
    from snapscribe.extract.vision import SUGGESTED_MODELS

    console.print("[dim]vision models worth benchmarking: " + ", ".join(SUGGESTED_MODELS) + "[/]")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover
        err_console.print("interrupted")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
