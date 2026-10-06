# Lessons this calibration already paid for

Each of these cost a wrong turn. Read them before proposing anything.

## Structure before appearance

The synthetic-trained YOLO went from 22% to 82% names right on the labelled real set
because of two STRUCTURAL fixes, found on 2026-09-24:

- the clinical lead order per number of columns (the generator used to spread a flat list
  over the page, an arrangement no printer makes);
- the lead name printed ABOVE the trace, as the INC's printers do (95% of the labelled
  set), instead of always below.

An earlier calibration (branch `feat/distribution-params`, `documentation/CALIBRACAO.md`
and `ACHADO_CADEIA_CALIBRACAO.md`) matched 71 global image statistics with populations and
chained bisection. It reached 3 to 12 of the 71 attributes and saw neither cause. Global
statistics do not see where things are on the page.

So, in order: channel and layout, lead order, name vocabulary and position, panels and
page furniture; only then paper, grid, trace, noise.

## The task decides

Matching image statistics is a necessary condition, never the result. The result is a YOLO
trained only on the synthetic set and scored on real pages (`eval_task.py`). Do not report
a calibration as an improvement on the strength of the profile table or the contact sheet.

## No attribution to realism groups yet

The v2 recipe switched seven realism groups on together. The ablation variants
(`batch_ptbxl_inc_exp_*.yaml`, 1500 pages each, one group at a time) exist as recipes but
were **never rendered or trained** (as of 2026-10-02). The only causal evidence is the
2026-09-24 controlled test of lead order and name position. Do not say that `inc_paper`,
`scan_look`, `handwriting` and so on "gave" any part of the gain, and do not rank them by
importance. If the user wants that ranking, the ablations are the way, and they are the
user's to start.

## A lot is a mixture

One archive holds pink strips, orange and white CLB sheets, phone photos, other machines.
A pooled median describes a page that does not exist. Name the populations in the audit
and work population by population.

The generator draws its options independently. Traits that go together in a population
(pink strip AND flatbed scan AND names above) therefore need a YAML per population, or a
realism group that ties them, not a set of independent probabilities.

## Coupled options do not converge one at a time

Tuning one parameter per pass undid the previous one: `white_balance` cancelled the
`grid_ink_ratio` fit (`ACHADO_CADEIA_CALIBRACAO.md`). In the chain, exposure and
white_point decide the blowout together, saturation delivered depends on exposure, and
highlight clipping weakens the white balance. Change a group of coupled values together
and re-measure the probe lot; do not bisect them in sequence.

## Small n and duplicates

- The labelled set has 66 valid images: the 95% interval on its accuracy is about
  +-6 points. A difference inside it is not progress. `eval_task.py` bootstraps over images
  and pairs the comparison.
- The same exam appears more than once: 300 and 600 dpi scans of one sheet, several photos
  of one sheet, one image saved twice. Count exams, not files, or one printer format is
  over-weighted. Exact copies fall to the pixel hash; same-exam copies to the date-time
  stamp printed in the footer. ORB matching does NOT work for this: every sheet of a
  printer model carries the same printed template and matches every other.

## Box convention: centre metric, not IoU

In the labelled real set the lead box takes the printed name in; a synthetic box does not.
IoU and mAP therefore score the convention. Compare with "right name and centre of the
kept box inside the labelled box".

## Orientation first

Scans arrive rotated by 90 or 270 degrees, in either direction, and the canvas shape does
not tell (a portrait canvas can hold an upright page). Neither YOLO scores orientation
usefully. Every structural measure on a sideways page is noise, so the orientation is
decided, by looking, before anything else is measured.

## What the measures can and cannot do

- "Same code on both sides" means measures that need no ground truth. Anything that reads
  `plotted_pixels` (the trace-anchored width in `trace_anchor.py`, the label geometry in
  `checks.py`) is a check of the synthetic side only.
- The grid scale from `bootstrap_scale` alone failed or landed on a harmonic on about a
  third of both real and synthetic pages. The rectifier's lattice is right to 0.2% when it
  is not a harmonic; `scale_suspect` flags the harmonics by the implied page size.
- An automatic lead-name position measure (text boxes against the dark-pixel row profile)
  agreed with the JSON truth on 5 of 7 synthetic pages: not good enough to drive weights.
  The audit reads it by eye.
- The label-free proxy follows accuracy (Spearman +0.82) with `best_v2.pt` on n=66. It is
  validated for that model only.

## Look at the page the way the detector does

The detector does not see the scan: Ultralytics shrinks it to its input size (1440 px for
`best_v2.pt`) with plain linear interpolation, no antialiasing. A fine grid aliases there
into moire bands that are invisible at native pixels and in a smoothed thumbnail - the
first audit of the labelled set missed its moire page for exactly that reason. The audit
composite therefore leads with the page as the detector receives it, with the boxes the
pipeline keeps drawn on it.

Moire also depends on the input SIZE. The labelled set's "moire" page shows strong beat
bands at 1024 px (nb 4.3's training default) and 800 px, and faint ones at 1440 px
(`best_v2.pt`). The generator's pages alias too when shrunk, so compare at MATCHED
resolution: that page (a ~355 dpi frame) scored 1.87 at 1024 px; synthetic pages rendered
at 350-420 dpi scored 0.53-2.42 (median 0.89, n=11). High, but inside what the training set
already holds - and its detector view shows boxes shifted by a row, a more likely cause of
its failures. So: always list moire, with the matched-resolution synthetic numbers beside
it; call it a generator gap only when the real pages exceed the synthetic ones at the same
dpi, or when the moire was born in the capture (a screen photographed, a halftone copy) and
shows at native pixels.

## The generator's RNG

`randomize:` consumes ONE per-record stream in ALPHABETICAL key order, and a `choice` list
of a different length spends a different number of draws. A new key there, or a list of
another length, re-rolls every key that sorts after it. New draws take a stream of their
own, as the realism groups do (one draw per group, in the fixed order of `GROUPS`; a new
group goes at the END).

## Privacy

The scans carry handwritten patient names, and printed ones in the header. Everything this
skill writes stays on this machine. Nothing with a real page goes to an Artifact, a chat
attachment or a repository unless the names are pixelated first and the user asked.

## Operation

- A page at 350-600 dpi is about 47 MB and the render peaks at about 12 GB of RAM on
  average, 21.6 GB at 598 dpi. A run that reaches swap dies: that killed a
  `ptbxl_synthetic_4000` run. One render at a time, `max_workers: 1` at these resolutions.
- Always a NEW output directory. The runner writes flat; `ptbxl_synthetic_4000` was
  overwritten once by a run pointed at it.
- AMD GPU (ROCm): an inference next to a training run brings both down. Everything here
  runs on CPU.
- Long runs log to `~/.cache/inc-ecg-generator-run/`, never to a session scratch
  directory, which is deleted with its processes.
