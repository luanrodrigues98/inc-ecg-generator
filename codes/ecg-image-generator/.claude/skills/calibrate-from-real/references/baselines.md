# Where things stand

Aggregate reference numbers only. Per-image findings of the labelled set are deliberately
not written here: that set is the blind test of this skill, and a list of its hard images
would let an audit "find" them without looking.

## Repositories and environments

| | path | python |
|---|---|---|
| generator | `/home/luan-rodrigues/faculdade/inc-ecg-generator/codes/ecg-image-generator` (branch `upstream-baseline`) | `./.venv310/bin/python` (3.10 only) |
| digitizer | `/home/luan-rodrigues/faculdade/inc-ecg-digitizer` | `poetry run python` (3.12) |

- Measurement scripts (everything in `scripts/` except `preflight.py`) run in the
  digitizer's Poetry environment, on CPU. `preflight.py` and the renders run in the
  generator's venv.
- Models, in the digitizer's `models/`: `best_v2.pt` (current lead YOLO-OBB, 12 classes,
  trained 2026-09-29 on the v2 synthetic set only), `best.pt` (the previous one),
  `best_yolo_obb.pt` (trained on labelled real data: a CEILING, never a baseline to beat
  with more real data), `rectifier_400dpi.pth`.
- Frozen test sets, in the digitizer's `datasets/`: `dataset_label/` (68 labelled real
  ECGs, 66 valid) and `cardio_s3_cache/` (104 readable unlabelled scans, orientation in
  `labels/cardio_orientation.csv`).
- Run outputs: `~/.cache/inc-ecg-generator-run/calib/<lot>-<YYYYMMDD>/`. Calibration log:
  `~/.cache/inc-ecg-generator-run/calib/calibration_log.jsonl`.

## Task numbers (2026-09-29; original images, CPU, corrected `get_image_boxes`)

| model | dataset_label: names right (centre) | Cardio: leads per page | Cardio: pages with 12 |
|---|---|---|---|
| `best.pt` (old synthetic) | 22% | 7.9 | 1% |
| `best_v2.pt` (synthetic v2, current) | 82% | 11.4 | 69% |
| `best_yolo_obb.pt` (real data, ceiling) | not valid (trained on it) | 11.4 | 73% |

- `best_v2.pt` on dataset_label: IoU >= 0.5 gives 58% and name swaps among located leads
  13% - the IoU figure is low because of the box convention, which is why the centre
  metric is the one to quote.
- The 95% interval on the dataset_label figure is about +-6 points (bootstrap over images).

## The label-free proxy

proxy = mean confidence of the leads `get_image_boxes` keeps x (leads kept / 12).

- With `best_v2.pt` on dataset_label (n=66): Spearman +0.82 with the true per-image
  accuracy; 9 of the 10 images with 6 or fewer names right are among the 10 worst by proxy.
  Reproduced by `check_proxy.py` on 2026-10-02 (+0.818).
- Confidence alone gives +0.71, lead count alone +0.76.
- Valid for that model. After a retrain, run `check_proxy.py --model <new>` before
  ranking a lot with it.

## What the v2 recipe already contains

Seven realism groups, each with a probability per record (`realism:` in
`batch_ptbxl_inc_v2.yaml`): `clinical_lead_order` 1.0, `lead_name_position` 1.0,
`inc_paper` 0.75, `inc_trace` 0.75, `lead_name_print` 0.9, `scan_look` 0.5,
`handwriting` 0.7. The values are a proposal; none was fitted against a trained model, and
the one-group ablation recipes (`batch_ptbxl_inc_exp_*.yaml`) were never run.

Measurements behind them (Cardio scans, 600 dpi), kept in the YAML comments: paper under
the grid (pink strip near-white, CLB orange salmon, CLB white), grid coverage 30-37% on the
pink strip, trace FWHM 0.11-0.19 mm (pink) and ~0.26 mm (CLB), capital height of the lead
names 1.8-2.4 mm, scan paper luminance median 245, noise ~2 levels, skew median 0.2 deg.
The scripts those came from are in `~/.cache/inc-ecg-generator-run/` (`checks.py`,
`trace_anchor.py`, `dpi_metrics.py`, `compare_labels.py`, `contact.py`).

## Cost of a render

- Photo-chain pages about 1.66e-4 x dpi^2 s, scan pages about half: measured ~14 s a scan
  page at 400 dpi in a probe; budget 30 s a page at 350-600 dpi.
- PNG about 1.764e-4 x dpi^2 + 6.2 MB, plus 1 MB of sidecars: ~47 MB a page at 350-600 dpi,
  ~140 GB for 3000 pages.
- Peak RAM 21.6 GB at 598 dpi (supersample 2, one worker), growing with dpi^2: ~10 GB at
  400 dpi. The machine has 30 GB.
- `profile_pages.py` with the default `--scale rectify`: about 5 s a page on CPU with 8
  threads (up to 30 s when other jobs share the CPU).
- `rank_proxy.py`: about 0.2 s a page; `intake.py scan` a few seconds for 70 pages.
- The audit is the expensive step: one composite image read per page.

## Known state of the synthetic side

- `~/.cache/inc-ecg-generator-run/calib/profile_synth_inc_v2.csv`: 25 pages of
  `ptbxl_synthetic_inc_v2`, of which 3 have a suspect grid scale and 2 none. With n=25 the
  synthetic p5/p95 are close to its min/max: use `--synth-filter` to compare against the
  matching kind of synthetic page.
- `ptbxl_synthetic_inc_v2` stopped on 2026-09-26 at record 02897: `02897_lr-0.json` and
  `02897_lr.dat` are empty and its PNG is 11550 px wide, about twice the expected size
  (most likely left at the supersampled render).
  About 2896 pages, not 3000. `profile_pages.py` skips pages whose JSON does not parse.
