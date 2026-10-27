# SnapScribe

Transcribe chat conversations from smartphone screenshots — both native device
screenshots and photos of a screen taken with another camera — into a clean CSV,
using a classical OCR engine cross-checked against an optional lightweight
vision LLM.

## Status

Phases 0–4 of the roadmap are implemented:

| Phase | Scope | State |
|---|---|---|
| 0 | `uv`/`ruff`/`pytest` setup, ground-truth corpus + scorer | done |
| 1 | Data model, ingest, resumable pipeline skeleton | done |
| 2 | Screenshot normalization, bubble layout detection, debug overlays | done |
| 3 | YAML profiles, pydantic validation, fingerprint auto-detection | done |
| 4 | Dual-path extraction (RapidOCR + optional vision model) | done |
| 5 | Verification / review queue | partial (wired; thresholds untuned) |
| 6 | Timestamps, stitching, CSV export | partial (works end to end) |
| 7–9 | GUI, photo registration, packaging | not started |

Phases 5 and 6 are present because the pipeline needs them to produce a CSV at
all. They work, but the review thresholds have not been tuned — on the synthetic
corpus almost nothing is wrong, so there is nothing to tune them against yet.
That needs real screenshots.

### Accuracy on the synthetic corpus (OCR only)

| set | recall | precision | CER | CER ignoring emoji | sender | side |
|---|---|---|---|---|---|---|
| whatsapp_light | 100% | 100% | 1.9% | 0.5% | 100% | 100% |
| whatsapp_dark | 100% | 100% | 2.5% | 1.0% | 100% | 100% |
| group_light | 100% | 100% | 7.0% | 0.4% | 100% | 100% |
| group_dark | 100% | 100% | 7.8% | 1.1% | 100% | 100% |
| media_light | 100% | 100% | 16.1% | 0.9% | 100% | 100% |
| media_dark | 100% | 100% | 11.1% | 1.1% | 100% | 100% |
| tablet_light | 100% | 100% | 1.1% | 0.8% | 100% | 100% |

The gap between the two CER columns is the emoji policy: classical OCR cannot
read emoji, so each becomes `[emoji]` and counts as a character error.

These screenshots are rendered by code in this repo from the colours the
profiles declare, so profile detection is graded on its own homework and there
are no real fonts, wallpapers or compression artefacts. Good scores here mean
the pipeline is wired up correctly — not that it works on your phone. See
`docs/TESTING.md`.

## Install

```bash
uv venv
uv pip install -e ".[ocr]" --group dev     # add ,vision for the LLM path
```

`[ocr]` pulls RapidOCR + onnxruntime (~80 MB incl. models, downloaded on first
run). Without it everything still runs — the OCR stage just yields empty text,
which is enough to test layout and the pipeline wiring.

## Quick start

```bash
# 1. Build the synthetic ground-truth corpus (no phone required)
snapscribe corpus build

# 2. Look at what layout detection found
snapscribe layout corpus/generated/whatsapp_light --overlay debug/

# 3. Full run
snapscribe run corpus/generated/whatsapp_light -o out/light.csv --profile auto

# 4. Score it against ground truth
snapscribe corpus score out/light.csv corpus/generated/whatsapp_light/ground_truth.csv
```

```bash
# What the segmenter actually saw — the most useful artefact when tuning
snapscribe layout ~/shots/jonas --overlay debug/jonas

# Which app/theme is this, and why?
snapscribe profile detect ~/shots/jonas

# What is it unsure about?
snapscribe review sessions/jonas.json
```

See `docs/TESTING.md` for the full test plan, including how to test against your
own screenshots with a WhatsApp export as ground truth.

## Design notes

* **Stages are independent.** Each stage reads and writes the `Session`, which
  serialises to JSON, so any stage can be rerun without redoing the others
  (`--from layout`). The GUI planned for phase 7 consumes the same artifacts.
* **Two extractors, one interface.** `Extractor.extract(crop) -> ExtractResult`.
  OCR and vision run on the *same bubble crops*, so their outputs line up
  one-to-one and can be diffed per message.
* **Profiles are data, not code.** App-specific geometry, colours, timestamp
  formats and fingerprints live in YAML validated by pydantic. Any value can be
  overridden from the command line with `--set section.key=value`.
* **The corpus is the measurement.** `snapscribe corpus build` renders
  screenshots from a WhatsApp export, so ground truth is exact and every metric
  — character error rate, messages found vs. missed, sender and timestamp
  accuracy — is computable from day one. The same scorer reads a *real* export,
  so the day you have real screenshots the measurement does not change.

## Layout, in four steps

1. Rows that are pixel-identical across screenshots are chrome; the runs
   touching the top and bottom edges are the header and footer.
2. Inside the chat area, every pixel far enough from the background colour is
   content. Morphology merges text into blobs; connected components are bubbles.
3. Side comes from position, with bubble fill colour as the second opinion.
4. A bubble whose cut edge is flat was truncated by the viewport — it is dropped
   here and recovered whole from the neighbouring screenshot.
