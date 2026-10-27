# How to test SnapScribe (phases 0–4)

Two things are worth your time here, in this order:

1. **Ten minutes on the synthetic corpus** — confirms the plumbing works on your
   machine and gives you a number to compare against later.
2. **An hour on your own screenshots** — the only test that tells us whether
   this works on real apps. Everything below is in service of that.

---

## 0. Setup

```bash
git clone <repo> && cd SnapScribe
git checkout claude/vigilant-bell-r8jd6q

uv venv
uv pip install -e ".[ocr]" --group dev
source .venv/bin/activate     # or prefix commands with .venv/bin/
```

`[ocr]` pulls RapidOCR + ONNX Runtime (~80 MB including the models). Check what
the install found:

```bash
snapscribe engines
```

`ocr` should say **yes**. `vision` will say no until you set up Ollama (§5).

---

## 1. Ten-minute smoke test

```bash
snapscribe corpus build                       # renders 7 screenshot sets
snapscribe corpus list
```

Open a couple of PNGs under `corpus/generated/` and look at them — they are
synthetic WhatsApp screenshots with light/dark themes, a group chat, emoji,
media bubbles and a day boundary.

```bash
# What the segmenter sees. Open these; they are the most useful artefact here.
snapscribe layout corpus/generated/group_dark --overlay debug/group_dark

# Full run + score
snapscribe run corpus/generated/whatsapp_light -o out/light.csv
snapscribe corpus score out/light.csv corpus/generated/whatsapp_light/ground_truth.csv
```

### What you should see

Scores on this machine, all seven sets, OCR only:

| set | recall | precision | CER | CER ignoring emoji | sender | side |
|---|---|---|---|---|---|---|
| whatsapp_light | 100% | 100% | 1.9% | 0.5% | 100% | 100% |
| whatsapp_dark | 100% | 100% | 2.5% | 1.0% | 100% | 100% |
| group_light | 100% | 100% | 7.0% | 0.4% | 100% | 100% |
| group_dark | 100% | 100% | 7.8% | 1.1% | 100% | 100% |
| media_light | 100% | 100% | 16.1% | 0.9% | 100% | 100% |
| media_dark | 100% | 100% | 11.1% | 1.1% | 100% | 100% |
| tablet_light | 100% | 100% | 1.1% | 0.8% | 100% | 100% |

The gap between the two CER columns **is** the emoji policy: classical OCR
cannot read emoji, so each one becomes `[emoji]` and counts as a character
error. The right-hand column is how well the text itself is being read.

Reproduce the whole table:

```bash
for s in whatsapp_light whatsapp_dark group_light group_dark media_light media_dark tablet_light; do
  grp=""; case "$s" in group_*) grp="--group";; esac
  snapscribe run corpus/generated/$s -o out/$s.csv $grp -q >/dev/null
  printf "%-15s" "$s"
  snapscribe corpus score out/$s.csv corpus/generated/$s/ground_truth.csv --worst 0 \
    | grep -E "recall|^CER|sender "
done
```

**Be sceptical of these numbers.** The corpus is rendered by code in this repo
using the same colour palette the profiles declare, so profile auto-detection is
graded on its own homework, and nothing here has real font rendering, JPEG
artefacts, a wallpaper or a photo of a screen. It proves the pipeline is wired
up correctly. It does not prove the app works. That is §3.

---

## 2. Automated tests

```bash
pytest                 # ~100 tests, about a minute
pytest -m "not ocr"    # skip the ones needing the OCR models
ruff check . && ruff format --check .
```

The end-to-end tests in `tests/test_pipeline.py` assert accuracy *floors*
(recall 100%, CER < 10%) against a small rendered corpus, so a regression in any
stage fails the build rather than silently degrading output.

---

## 3. The test that matters: your own screenshots

### Capture

1. Pick a conversation in WhatsApp (or any app with chat export).
2. Scroll to the start and take screenshots down to the end, **overlapping by
   roughly a third of a screen**. 5–15 screenshots is plenty.
3. In the same chat: ⋮ → More → Export chat → **Without media**. Mail it to
   yourself. That `.txt` is the ground truth.
4. Put the screenshots in one folder, say `~/shots/jonas/`.

Worth capturing, in separate folders, if you can:

- the same conversation in light **and** dark mode
- a group chat
- a conversation with emoji, photos and voice notes
- a chat in a *different app* (Signal, Telegram, Messenger, iMessage) — I expect
  this to need a new profile, and I want to see where it breaks
- a conversation in a language other than English

### Run it

```bash
# A. Which app does it think this is, and why?
snapscribe profile detect ~/shots/jonas

# B. Look at what it found. This is the step people skip; don't.
snapscribe layout ~/shots/jonas --overlay debug/jonas
open debug/jonas/*_layout.png

# C. Full run
snapscribe run ~/shots/jonas -o out/jonas.csv -s sessions/jonas.json --overlay debug/jonas

# D. Score against the real export
snapscribe corpus score out/jonas.csv ~/Downloads/WhatsApp\ Chat\ with\ Jonas.txt

# E. What it is unsure about
snapscribe review sessions/jonas.json
```

### Reading the overlay

| colour | meaning |
|---|---|
| green box | incoming (left-hand) bubble |
| orange box | outgoing (right-hand) bubble |
| purple box | system message or date separator |
| pink lines | rows inside a bubble treated as a photo, not text |
| red dashed | dropped — clipped by the viewport, too small, or too wide |
| grey | header / footer chrome that was cut away |
| teal | the detected chat area |

Judge it in this order:

1. **Is the chat area right?** The teal box should start below the app bar and
   end above the composer. If it swallows the header, chrome detection failed —
   usually because the screenshots are too similar to each other.
2. **Is every message boxed exactly once?** Two boxes on one bubble, or one box
   over two bubbles, is the thing to report.
3. **Are the sides right?** Green on the left, orange on the right.
4. **Are clipped bubbles dropped?** Partial bubbles at the top and bottom should
   be red-dashed; they will be whole in the neighbouring screenshot.

### If it looks wrong, try these before reporting

```bash
# bubbles missed entirely → the mask is too strict (most common on dark themes)
snapscribe layout ~/shots/x --overlay debug/x --set segmentation.bg_tolerance=10

# two stacked bubbles merged into one box
snapscribe layout ~/shots/x --overlay debug/x --set segmentation.row_gap_px=3

# everything merged into one giant box → the close kernel is too wide
snapscribe layout ~/shots/x --overlay debug/x --set segmentation.close_kernel='[21, 7]'

# wrong app/theme detected → say so explicitly
snapscribe run ~/shots/x -o out/x.csv --profile whatsapp_dark

# group chat (sender names printed inside bubbles)
snapscribe run ~/shots/x -o out/x.csv --group

# a participant who only speaks once loses their name; name them
snapscribe run ~/shots/x -o out/x.csv --group --set senders.known_senders='["Mira","Tobias"]'

# screenshots taken long after the conversation: anchor what "Today" means
snapscribe run ~/shots/x -o out/x.csv --anchor-date 2024-03-12
```

`snapscribe profile show whatsapp_light` prints every value you can override.

---

## 4. Specific things I would like checked

These are the parts I am least sure about on real data:

1. **Chrome detection.** It assumes the header/footer are pixel-identical across
   screenshots. A changing "typing…" indicator, a scroll-to-bottom button or an
   unread-count badge may break it. Check the teal box.
2. **Dark themes.** The gap between a dark background (#0B141A) and an incoming
   bubble (#202C33) is ~25 units; JPEG or screen-photo noise could swallow it.
3. **Clipped bubbles.** The detector assumes rounded corners — a bubble with a
   flat cut edge near the viewport edge was truncated. Square-cornered apps need
   `--set layout.rounded_bubbles=false`.
4. **Wallpapers.** A patterned chat wallpaper is not in the corpus at all, and I
   expect it to hurt. This is my top suspect for real-world failure.
5. **Emoji.** Only *saturated* emoji are detected (❤️ yes, ☁️ no — a white
   cloud looks like text). Check how many go missing.
6. **Stitching.** With a big scroll step, consecutive screenshots may share no
   complete message and nothing can be de-duplicated. That is reported and is
   normal; what would be a bug is a *duplicated* or *dropped* message.
7. **Timestamps.** Dates come from the separators ("TODAY"/"YESTERDAY") resolved
   against when the screenshots were taken. Check the dates, not just the times.

---

## 5. The vision path (optional)

The second extractor is unproven — I built the interface and the benchmark, not
a verdict on which model to use. There is no GPU here, so I have not run it.

```bash
# install ollama, then
ollama pull qwen2.5vl:3b
snapscribe engines                      # vision should now say yes

snapscribe run ~/shots/jonas -o out/jonas_dual.csv --vision
snapscribe review sessions/jonas.json   # engine disagreements are flagged here
```

Benchmark candidates against ground truth rather than picking one:

```bash
snapscribe bench corpus/generated/media_light \
  --engines ocr,vision --models qwen2.5vl:3b,moondream,minicpm-v
```

It prints recall, CER, exact-match and wall-clock per engine. The question to
answer is whether the vision path earns its runtime — expect it to be 10–50×
slower than OCR per bubble. It should win on emoji and on blurry photos; if it
does not win on anything, we drop it and keep the interface.

---

## 6. What to send back

Most useful, in order:

1. The **debug overlays** for anything that looks wrong — one picture explains
   more than a paragraph.
2. The **score output** plus the `.csv` and the export `.txt` for real
   conversations.
3. The **session file** (`sessions/*.json`) — it holds every stage's output, so
   I can rerun one stage without your screenshots.
4. Which `--set` overrides you needed to make an app work. That is the raw
   material for the next built-in profile.

---

## Known limitations at this stage

| | |
|---|---|
| Emoji | Classical OCR replaces them with `[emoji]`; only saturated ones are detected at all. The vision path reads them properly. |
| Attachments | A photo or video bubble is detected by colour and marked `image`; video/GIF are not distinguished. Audio is detected only from a bare duration. |
| Single screenshots | Chrome detection needs two or more; with one, profile fractions are used and the chat area is approximate. |
| Photos of a screen | Phase 8. The quality checks exist, but there is no perspective correction yet. |
| GUI | Phase 7, not started. |
| Profiles | WhatsApp light/dark plus a permissive `generic` fallback. Other apps will need their own. |
