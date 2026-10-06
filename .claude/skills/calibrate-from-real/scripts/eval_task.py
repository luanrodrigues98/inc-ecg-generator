#!/usr/bin/env python
"""Step 6 - the measure that decides: a YOLO trained on synthetic pages, scored on real ones.

    eval_task.py --model models/best_v2.pt [--model models/<new>.pt ...]
                 [--note "batch_ptbxl_calib_x.yaml, 3000 pages"] [--no-cardio] [--no-log]

Two frozen test sets of the digitizer, original images, CPU:

  dataset_label   66 labelled real ECGs (img202 and img218 carry wrong labels and are left
                  out; img213 and img214 are the same pixels and count as one cluster).
                  Metric: the kept box of each labelled lead has the right NAME and its
                  CENTRE inside the labelled box. Not IoU/mAP: here the labelled box takes
                  the printed name in and a synthetic one does not, so IoU scores the box
                  convention, not the detector.
  Cardio          104 unlabelled scans (datasets/cardio_s3_cache): leads kept per page,
                  share of pages with all 12, median confidence.

The dataset_label figure comes with a 95% interval from a bootstrap over IMAGES (leads of
one page fail together, so resampling leads would be far too optimistic). With n=66 the
interval is about +-6 points: when two models are given, the difference is bootstrapped
on the same resamples (paired), and a difference whose interval spans 0 is not progress.

The labelled set is for MEASURING only - never train on it, never tune a YAML against a
single image of it. That is why no per-image result is printed or logged here. Every run is appended to the calibration log
(~/.cache/inc-ecg-generator-run/calib/calibration_log.jsonl).
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from _common import (DIGITIZER, LEAD_NAMES, RUN_ROOT, cardio_orientation, detect_leads, load_yolo,
                     read_image, upright)

EXCLUDED = {"img202", "img218"}
SAME_PIXELS = {"img214": "img213"}
REFERENCE = "reference (2026-09-29): best.pt 22% / 7.9 leads / 1% with 12; best_v2.pt 82% / 11.4 / 69%"

parser = argparse.ArgumentParser()
parser.add_argument("--model", action="append", required=True)
parser.add_argument("--note", default="")
parser.add_argument("--no-cardio", action="store_true")
parser.add_argument("--no-log", action="store_true")
parser.add_argument("--resamples", type=int, default=2000)
args = parser.parse_args()


def dataset_label(model):
    """Per image cluster: [names right, leads labelled]."""
    per_image = {}
    for path in sorted((DIGITIZER / "datasets" / "dataset_label").glob("*.png")):
        if path.stem in EXCLUDED:
            continue
        bgr, _ = read_image(path)
        h, w = bgr.shape[:2]
        leads, _result = detect_leads(model, bgr)
        hits = total = 0
        for line in path.with_suffix(".txt").read_text().splitlines():
            values = line.split()
            if len(values) != 9:
                continue
            total += 1
            truth = np.array(values[1:], float).reshape(4, 2) * [w, h]
            kept = leads.get(LEAD_NAMES[int(values[0])].lower())
            if kept is not None:
                cx, cy = kept[0].mean(0)
                hits += truth[:, 0].min() <= cx <= truth[:, 0].max() and truth[:, 1].min() <= cy <= truth[:, 1].max()
        cluster = per_image.setdefault(SAME_PIXELS.get(path.stem, path.stem), [0, 0])
        cluster[0] += hits
        cluster[1] += total
    return per_image


def cardio(model):
    # keyed by the md5(key)[:8] prefix of the cache name, as the reference numbers were
    orientation = {name.split("__", 1)[0]: rot for name, rot in cardio_orientation().items()}
    n_leads, confs = [], []
    for path in sorted((DIGITIZER / "datasets" / "cardio_s3_cache").glob("*__*.png")):
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue  # the PNG truncated at the source: left out of the reference numbers too
        leads, result = detect_leads(model, upright(bgr, orientation.get(path.name.split("__", 1)[0], 0)))
        n_leads.append(len(leads))
        raw = result.obb.conf.cpu().numpy() if result.obb is not None else result.boxes.conf.cpu().numpy()
        confs.append(float(np.median(raw)) if len(raw) else np.nan)
    return np.array(n_leads), np.array(confs)


rng = np.random.default_rng(0)
results, tables = [], {}
for name in args.model:
    path = Path(name) if Path(name).is_absolute() else DIGITIZER / name
    model = load_yolo(path)
    per_image = dataset_label(model)
    keys = sorted(per_image)
    table = np.array([per_image[k] for k in keys], float)
    tables[path.name] = (keys, table)
    picks = rng.integers(0, len(keys), size=(args.resamples, len(keys)))
    boot = table[picks, 0].sum(1) / table[picks, 1].sum(1)
    out = dict(model=path.name, images=len(keys), names_right=float(table[:, 0].sum() / table[:, 1].sum()),
               ci95=[float(v) for v in np.percentile(boot, [2.5, 97.5])])
    if not args.no_cardio:
        n_leads, confs = cardio(model)
        out.update(cardio_pages=int(len(n_leads)), cardio_leads=float(n_leads.mean()),
                   cardio_all12=float((n_leads == 12).mean()), cardio_conf=float(np.nanmedian(confs)))
    results.append(out)
    print("ok %s" % path.name, flush=True)

print("\n%-28s %-28s %s" % ("model", "dataset_label names right", "Cardio leads/page | pages with 12 | conf"))
for out in results:
    cardio_text = "-" if args.no_cardio else "%.1f | %.0f%% | %.2f (n=%d)" % (
        out["cardio_leads"], 100 * out["cardio_all12"], out["cardio_conf"], out["cardio_pages"])
    print("%-28s %-28s %s" % (out["model"], "%.1f%% (95%% CI %.1f-%.1f, n=%d)" % (
        100 * out["names_right"], 100 * out["ci95"][0], 100 * out["ci95"][1], out["images"]), cardio_text))
print(REFERENCE)

differences = []
names = list(tables)
for other in names[1:]:
    (keys, a), (_keys, b) = tables[names[0]], tables[other]
    picks = rng.integers(0, len(keys), size=(args.resamples, len(keys)))
    diff = b[picks, 0].sum(1) / b[picks, 1].sum(1) - a[picks, 0].sum(1) / a[picks, 1].sum(1)
    low, high = np.percentile(diff, [2.5, 97.5])
    point = b[:, 0].sum() / b[:, 1].sum() - a[:, 0].sum() / a[:, 1].sum()
    verdict = "better" if low > 0 else "worse" if high < 0 else "NOT distinguishable at n=%d" % len(keys)
    differences.append(dict(model=other, against=names[0], diff=float(point), ci95=[float(low), float(high)], verdict=verdict))
    print("%s - %s: %+.1f points (95%% CI %+.1f to %+.1f, paired by image) -> %s" % (
        other, names[0], 100 * point, 100 * low, 100 * high, verdict))

if not args.no_log:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with open(RUN_ROOT / "calibration_log.jsonl", "a") as handle:
        handle.write(json.dumps(dict(when=time.strftime("%Y-%m-%d %H:%M"), note=args.note, results=results,
                                     differences=differences)) + "\n")
    print("appended to %s" % (RUN_ROOT / "calibration_log.jsonl"))
