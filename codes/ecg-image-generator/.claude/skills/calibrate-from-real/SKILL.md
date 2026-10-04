---
name: calibrate-from-real
description: Calibrate the synthetic ECG image generator (inc-ecg-generator) against a directory of REAL, unlabelled ECG images - find what the generator still does not represent (layout, lead-name convention, paper and grid, scan vs photo, degradations), propose realism probabilities in a NEW batch YAML, write feature requests for what no option can draw, render a small probe lot, and leave the generate / train / evaluate commands ready. Use this whenever a new lot, batch or folder of real ECGs arrives, when the user asks to recalibrate or re-tune the generator, asks what the synthetic data is missing, asks why the synthetic-trained lead YOLO fails on some real pages, or wants real and synthetic ECG pages compared - even if the word "calibrate" never appears. Portuguese triggers too - "chegou um lote novo", "recalibre o gerador", "o que falta no sintetico", "por que o YOLO erra nessas imagens".
---

# Calibrate the generator from a lot of real ECG images

The lead-detection YOLO is trained ONLY on synthetic pages. Real lots arrive without
labels, and labelling each one does not scale, so the generator has to be brought to the
real pages instead. This skill takes a directory of real ECG images and answers: **in what
is the generator still not drawing these pages, and what should change?**

The centre of the work is DISCOVERING GAPS. Fitting numbers comes after, and only for the
gaps an existing option covers.

Answer the user in the language they write in (usually Portuguese). Files this skill
writes inside the generator repository are in English, the repository's policy.

## Ground rules

These hold for the whole run. Each has cost something already.

- **The user starts trainings and long renders.** Leave the commands ready and stop. The
  only render this skill runs is a probe lot that `preflight.py --probe` approves (at most
  24 pages, 400 dpi, one worker). No AWS command of any kind.
- **Synthetic-only training.** Real labelled data (`dataset_label`) is for measuring and
  stays frozen. Never propose training on it, never tune a value against one of its images.
- **Never overwrite.** Existing batch YAMLs and existing dataset directories are read-only:
  a calibration writes a NEW YAML that `extends:` the recipe and renders into a NEW
  directory. `ptbxl_synthetic_4000` was destroyed once by a run pointed at it.
- **Patient pages stay local.** Scans carry patient names, printed and handwritten. Sheets
  and views are written under `~/.cache/` with `_LOCAL` in the name. Nothing with a real
  page goes to an Artifact, an upload or a repository.
- **CPU only, one heavy job at a time.** On this AMD GPU an inference next to a training
  run brings both down, and a render that reaches swap dies.
- **No commit unless asked.** And before writing anything in the generator repository,
  check `git status` there: another session may be editing it.
- **Do not read label files during a run.** If the lot directory happens to hold `.txt`
  or `.json` annotations, leave them closed: the point is a method that works without them.

## Setup

```bash
GEN=/home/luan-rodrigues/faculdade/inc-ecg-generator/codes/ecg-image-generator
DIG=/home/luan-rodrigues/faculdade/inc-ecg-digitizer
SK=$GEN/.claude/skills/calibrate-from-real/scripts
LOT=<directory of real images>
RUN=~/.cache/inc-ecg-generator-run/calib/<lot-name>-<YYYYMMDD>     # new for every run
export PYTHONDONTWRITEBYTECODE=1    # no __pycache__ left inside the generator repository
```

Every script except `preflight.py` runs in the digitizer's Poetry environment:
`cd $DIG && poetry run python $SK/<script>.py ...`. Renders and `preflight.py` run in the
generator's own venv: `cd $GEN && ./.venv310/bin/python ...`.

Read `references/lessons.md` now, before step 0. It is short, and every item in it is a
mistake this calibration already made once. `references/baselines.md` has the current
numbers, models and paths.

## The workflow

Work through the steps in order. Each ends with something written under `$RUN`, so a run
can be resumed and the user can check any step. When the user asks only for a diagnosis
("what is missing?", "is this format covered?"), steps 0 to 4 are the whole job: no YAML,
no render. "Recalibrate" means all of 0 to 6.

Order of the heavy jobs (one model at a time): `rank_proxy.py` (step 1), then
`audit_views.py` (step 2 - it runs the detector too), then start `profile_pages.py`
(step 3) in the background and read the audit composites while it runs. Compare profiles
only after the audit, once repeats of an exam are recorded.

Everything under `$RUN` shows real pages and stays on this machine; the sheets meant for
looking at carry `_LOCAL` in their names as a reminder.

Language: the report to the user, `gaps.md` and the audit notes in the user's language;
feature requests in English, because they go to a session in the generator repository.

### 0. Intake and hygiene

```bash
poetry run python $SK/intake.py scan $LOT $RUN
```

Reports files that do not decode (truncated PNGs are recovered and flagged), exact copies
(same decoded pixels under another name), and writes numbered thumbnail sheets.

**Orientation, before anything else.** Read the `orientation_sheet_NN_LOCAL.jpg` sheets
and decide, for every page, how many degrees CLOCKWISE it must turn to read upright.
Scans come turned 90 or 270 degrees in either direction; the canvas shape does not tell,
and no model here does it for you. Upright means: traces run left to right, lead names
read normally, the calibration pulse stands on its base. When a sheet tile is too small
to tell 0 from 180, read that page's `thumbs/<slug>.jpg`. If a page stays unclear, ask.

```bash
poetry run python $SK/intake.py orient $RUN "default=270;5=90;17,18=0"   # idx from the sheets
poetry run python $SK/intake.py check $LOT $RUN                          # then LOOK at upright_sheet_*
```

Only for files that keep their names from the digitizer's Cardio cache, `orient $RUN
cardio` takes the hand-made labels instead.

**Repeats of the same exam.** A lot counts exams, not files: the same sheet scanned at 300
and at 600 dpi, or photographed three times, is one page of evidence. Exact copies (same
pixels) are already dropped from every later step. For the rest:

```bash
poetry run python $SK/intake.py pairs $LOT $RUN                # candidates + pair_<a>_<b>_LOCAL.jpg
poetry run python $SK/intake.py same-exam $RUN "7=3;12=9"       # after LOOKING: #7 repeats #3
```

`pairs` proposes rescans of one sheet by lining up the dark ink; read each pair image and
keep only those whose traces agree beat for beat. It cannot find two photographs of one
sheet from different angles: those show up in the audit, where the exam's date and time
(printed in the footer on some printers, handwritten on others, absent on some) goes in
`exam_stamp`; record them with `same-exam` then, and rerun `audit_summary.py`. Do not try
feature matching (ORB and the like): every sheet of a printer carries the same printed
template and matches every other.

### 1. Rank without labels

```bash
poetry run python $SK/rank_proxy.py $LOT $RUN            # --model models/<other>.pt after a retrain
```

Runs the current YOLO as the pipeline does and orders the pages by
proxy = mean confidence of the kept leads x (leads / 12), worst first. With `best_v2.pt`
the proxy follows the true accuracy (Spearman +0.82 on the labelled set). After a
retrain, run `check_proxy.py --model ...` first: the validation is per model.

The ranking says where to look. It does not say why, and it is not a gap list: a page can
score low for a reason the generator already covers.

### 2. Visual audit

```bash
poetry run python $SK/audit_views.py $LOT $RUN --worst 12 --control 12
```

Picks the worst pages by proxy and a random control sample of the rest, and cuts views of
each (`$RUN/audit/<slug>/`). Read `audit.jpg`, one composite per page. Its top panel is
the page AS THE DETECTOR RECEIVES IT - shrunk to the model's input size without
antialiasing, with the leads the pipeline kept in green and the discarded boxes in red -
so you see both the page and what the detector made of it (missed leads, swapped names,
a box swallowing the next row, a calibration pulse taken for a lead). Moire only exists at
that scale. Below it: both ends of the lead rows (pulse on the left or the right, names,
device text), a native-pixel detail, header and footer. Open a single view only when the
composite leaves a field undecided. Never read the original files: scans run to 5000 x 6500 px.

Record one JSON line per page in `$RUN/audit.jsonl` following
`references/audit-checklist.md` - fields and vocabulary are fixed there. Append a few
pages at a time rather than all at the end: an audit can be interrupted, and
`audit_summary.py --partial` summarises what is already there.

The two groups do different jobs. The worst pages are where gaps are **discovered**. The
control sample is where their **frequency** is estimated, because the worst pages are a
biased sample by construction. Size the audit by the lot:

- up to ~70 pages: audit the whole lot (`--all`); every page is then a control page,
  and `audit_summary.py` still reports the worst pages by rank;
- larger: `--worst 12 --control 30` or more - twelve control pages only bound a share to
  +-20 points.

```bash
poetry run python $SK/audit_summary.py $RUN     # validates the JSONL, writes audit_summary.md
```

The summary gives, for every trait, how common it is and what it costs the detector
(median proxy and mean leads kept with and without it). Both halves feed a gap's priority.
The proxy does not see a lead given the wrong name or a box shifted by a row: for those,
the detector view of the pages is the evidence.

Two pages the audit suspects are one exam (same date and time, same beats) can be put side
by side with `intake.py view $RUN 4,37`; record confirmed repeats with `intake.py
same-exam` and rerun the summary.

### 3. Measured profile

```bash
poetry run python $SK/profile_pages.py $LOT $RUN/profile_real.csv --run-dir $RUN            # --sample N on big lots
poetry run python $SK/profile_pages.py $DIG/datasets/ptbxl_synthetic_inc_v2 $RUN/profile_synth.csv --synthetic --sample 25
poetry run python $SK/compare_profiles.py $RUN/profile_real.csv $RUN/profile_synth.csv --out $RUN/profile_table.md
```

The same code measures both sides: grid scale and physical page size, paper and grid
colour, grid coverage, colourfulness, paper noise, trace width and darkness, skew, and the
moire a shrink to the detector's input creates (`alias_*`; generator pages have it too:
compare it at matched resolution, `--synth-filter "truth_dpi < 420"` for a ~350-400 dpi lot). It is
slow (10-30 s a page: it runs the rectifier for the scale), so run it in the background
and sample large lots. The synthetic side only changes when the dataset does: if
`~/.cache/inc-ecg-generator-run/calib/profile_synth_inc_v2.csv` exists, use it instead of
profiling `ptbxl_synthetic_inc_v2` again, and save a new one there under the dataset's
name when you do profile one.

Structure is NOT in this profile: layout, lead order, name vocabulary and name position
come from the audit. That is deliberate. Measuring structure with the YOLO being improved
is circular, and the one automatic name-position measure tried was right on 5 of 7 pages.

Compare with `--run-dir $RUN`, so repeats of an exam count once. If the audit named more
than one population, write `$RUN/populations.csv` (`slug,population`, every profiled
page) and compare per population (`--populations ... --population NAME`), against the
matching synthetic pages (`--synth-filter "truth_palette == 7"`, `"truth_scan_look == 1"`). A pooled
comparison of a mixed lot is misleading. Populations for comparison should hold at least
~5 pages: keep the audit's fine names, and pool devices seen once or twice into one coarse
`other_devices` population here.

### 4. Gaps

Write `$RUN/gaps.md`. Structure before appearance: channel and layout first, then lead
order, name vocabulary and position, panels and page furniture, and only then paper,
grid, trace and noise.

For every way the real pages differ from what the generator draws:

- **What it is**, in one line, and which **population** it belongs to.
- **Evidence**: frequency in the control sample with its interval (from
  `audit_summary.md`), frequency among the worst pages, example pages by slug with the
  path of their views, and the profile attribute if one measures it. A gap without a page
  of this lot behind it is a guess: leave it out.
- **Class**, against `references/generator-options.md` and the current recipe:
  - **(a) an option exists** - propose the probability or range, from the measured
    frequency, and say which key.
  - **(b) nothing draws it** - a feature request, written from
    `references/feature-request.md` into `$RUN/feature_requests/`, for every (b) gap of
    medium priority or more; low ones stay in `gaps.md` as "watch". This skill does not
    implement generator code; it hands over evidence and a draft prompt.
- **Priority**: frequency in the control sample x how badly the detector does on those
  pages. State both.

Every trait the audit tagged that the generator cannot draw goes into a **watch** table at
the end of `gaps.md`, with its count and pages - singletons included, and also when its
cost cannot be told apart from other traits of the same pages. Folding a rare trait into
another gap's examples loses it: the next lot is how a watch item becomes a gap.

Then say what is NOT a gap: traits the lot shares with the synthetic set. When the lot is
in distribution - high proxy across the board, the detector view showing the leads found
and named, and nothing the generator cannot draw touching the leads, their names, the
rows or the layout - say so plainly and stop after step 4 with "no change proposed". Page
furniture away from the leads (punch holes, edge marks) is worth listing, not a reason to
change the recipe. Inventing a gap to have something to deliver is the failure mode here.

Known, already documented (b) gaps - today the US-letter-only page (`generator-options.md`)
- are reported once, as known, with this lot's frequency and with any detector cost seen
here. They do not by themselves make a lot out of distribution or block "no change
proposed"; write a feature request for one (or add to a previous run's) only when this lot
adds evidence of cost.

An (a) difference of appearance with no measurable cost in a lot that is otherwise in
distribution (a paper a little lighter, a trace a little thinner) is listed as low priority
with the option that would move it, and left for a recalibration that has a reason to run.

When the user picked the lot by format ("these are the pink strips"), its shares describe
the selection, not the archive: say so, and do not move any recipe probability from them.

`gaps.md` opens with a table - gap, population, control share (interval), worst share,
class, proposal, priority, example pages - and the detail follows.

### 5. Probe lot and comparison

Only when step 4 produced class (a) proposals.

One NEW YAML per population, copied from `assets/probe_template.yaml` to
`$RUN/probe_<population>.yaml`: it extends the current recipe, pins that population's
traits, and renders a few pages at 400 dpi into `$RUN/probe_<population>/`.

```bash
cd $GEN
./.venv310/bin/python .claude/skills/calibrate-from-real/scripts/preflight.py $RUN/probe_<pop>.yaml --probe
./.venv310/bin/python run_batch_from_config.py $RUN/probe_<pop>.yaml        # only on "OK", one at a time
```

`preflight.py` validates the YAML through the runner itself and refuses a tracked recipe,
a used output directory, too little RAM or disk, or a second render. Read the
`realism (p)` line it prints: a bare `realism:` key silently switches every group off.

Then measure the probe with the same code and look at it beside the real pages:

```bash
cd $DIG
poetry run python $SK/profile_pages.py $RUN/probe_<pop> $RUN/profile_probe_<pop>.csv --synthetic
poetry run python $SK/compare_profiles.py $RUN/profile_real.csv $RUN/profile_probe_<pop>.csv \
    --populations $RUN/populations.csv --population <pop> --out $RUN/profile_table_<pop>.md
poetry run python $SK/contact_sheet.py $LOT $RUN $RUN/probe_<pop> $RUN/contact_<pop>_LOCAL.png \
    --populations $RUN/populations.csv --population <pop>
```

Read the contact sheet yourself. The table says whether the ranges overlap; only the
sheet says whether a synthetic page could be taken for one of the real ones. If a coupled
set of values is off (exposure with white point, saturation with exposure), change them
together and render again - one at a time they undo each other.

One full YAML, not one per population: the training notebook reads a single directory,
and two batches over the same corpus render the same record names, so they cannot simply be
merged. Express the populations inside the one YAML through the realism probabilities and
palette weights. When a trait cannot be mixed into one batch (`random_bw`, a different
`full_mode`), say so in `commands.md` as a limitation rather than proposing two datasets.

How far to move a probability: toward the lot's share when the lot stands for what the
digitizer will be fed from now on; otherwise widen - add the missing trait - without
narrowing what the recipe already covers. If it is not clear which, ask the user.

The proposal for the full dataset is written last, as
`$GEN/batch_ptbxl_calib_<lot>_<YYYYMMDD>[_<pop>].yaml`: full size, the recipe's own dpi
range, a new `output_directory` under the digitizer's `datasets/`. It is the only file a
run adds to the generator repository. If a file of that name exists, tracked or not, it is
someone's earlier proposal: pick another name, never overwrite. Run
`preflight.py ... --full` on it; do not commit it.

### 6. Validation on the task

Matching the profile is a necessary condition. What decides is a YOLO trained on the new
synthetic set and scored on real pages. Write `$RUN/commands.md` with the three commands,
filled in, and hand it to the user - do not run the first two:

1. **Generate** - the line `preflight.py <full yaml> --full` prints (nohup, log under
   `~/.cache/inc-ecg-generator-run/`), with its cost estimate (disk, hours, peak RAM). A
   full lot at 350-600 dpi runs close to this machine's memory: pass the RAM warning on to
   the user word for word. If preflight refuses (disk, a used directory), say so; do not
   work around it.
2. **Train** - `$DIG/notebooks/4.3.train_yolo_obb_synthetic_val.ipynb`, started by the
   user. Name the three things to set for a new dataset: `ECG_OBB_SRC=<new dataset dir>`
   in the environment before Jupyter starts, `N_ESPERADO` (pages in the dataset) and
   `RUN_NAME` in the first code cell. `best_v2.pt` was trained at `imgsz=1440`. The
   notebook promotes the best checkpoint to `models/leads_obb_synthetic_val.pt`: copy it
   to a name of its own before the next run overwrites it.
3. **Evaluate** - after training. These two scripts read the frozen labelled set; that is
   their job, and the reason they belong here and never in steps 0-4 of a run:

   ```bash
   cd $DIG && poetry run python $SK/eval_task.py --model models/best_v2.pt --model models/<new>.pt \
       --note "<yaml>, <pages> pages"
   ```

   It scores both models on the frozen labelled set (names right by the centre metric,
   with a bootstrap interval over images and a paired difference) and on the Cardio
   scans, and appends the round to the calibration log. A difference whose interval spans
   zero is not progress, whatever its sign. Then rerun `check_proxy.py` with the new model
   and `rank_proxy.py` on the lot, to see whether the pages that motivated the change
   moved.

## What to hand the user

A short report, in their language, leading with the answer:

1. **Verdict** - the lot is in distribution / these N gaps, most important first.
2. **Hygiene** - files unreadable or truncated, duplicates, how orientation was decided.
3. **Gap table** - as in `gaps.md`: frequency with interval, class, proposal, priority,
   example pages.
4. **What was produced** - paths of `gaps.md`, the feature requests, the new YAML(s), the
   contact sheet(s), `commands.md`.
5. **What is not known** - sample sizes, populations seen once, measures that failed
   (pages without a usable grid scale), anything decided by eye.

Numbers carry their n. Say "3 of 12 control pages (25%, 90% interval 9-51%)", not "about
a quarter". Do not claim an improvement: until step 6 is run there is a proposal, not a
result.

## References

- `references/lessons.md` - what went wrong before and why the workflow is shaped this
  way. Read first.
- `references/audit-checklist.md` - the audit fields, their vocabulary, how to look.
- `references/generator-options.md` - what the generator can draw, key by key; how a new
  YAML is written; turning a frequency into a value.
- `references/baselines.md` - current numbers, models, paths, costs.
- `references/feature-request.md` - template for a class (b) gap.
- `assets/probe_template.yaml`, `assets/audit_schema.json`.
