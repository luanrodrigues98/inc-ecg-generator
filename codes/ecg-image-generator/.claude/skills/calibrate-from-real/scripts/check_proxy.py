#!/usr/bin/env python
"""Does the label-free proxy still follow the true accuracy? Run when the YOLO changes.

    check_proxy.py [--model models/best_v2.pt] [--out proxy_check.csv]   (takes ~2 min, prints at the end)

Uses the frozen labelled test set datasets/dataset_label (66 valid images; img202 and
img218 carry wrong labels). THIS SCRIPT READS THE LABELS: it is a check on the tool, never
a step of a blind lot. Reference: Spearman +0.82 with best_v2.pt, and 9 of the 10 images
with 6 or fewer names right among the 10 worst by proxy.

True accuracy of an image = share of its labelled leads whose kept box has the right name
and its centre inside the labelled box ("centre" metric: the labelled boxes include the
printed name and the synthetic ones do not, so IoU would punish a convention).
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from _common import DIGITIZER, LEAD_NAMES, detect_leads, load_yolo, proxy_score, read_image, write_csv

EXCLUDED = {"img202", "img218"}

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="models/best_v2.pt")
parser.add_argument("--out", default="")
args = parser.parse_args()

model_path = Path(args.model) if Path(args.model).is_absolute() else DIGITIZER / args.model
model = load_yolo(model_path)
rows = []
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
    rows.append(dict(image=path.stem, proxy=proxy_score(leads), n_leads=len(leads), right=hits,
                     labelled=total, accuracy=hits / max(total, 1)))

proxy = np.array([r["proxy"] for r in rows])
accuracy = np.array([r["accuracy"] for r in rows])
rho = spearmanr(proxy, accuracy).correlation
worst10 = {rows[k]["image"] for k in np.argsort(proxy)[:10]}
low = [r["image"] for r in rows if r["right"] <= 6]
print("\nmodel %s, n=%d" % (model_path.name, len(rows)))
print("names right (centre metric): %.1f%%" % (100 * sum(r["right"] for r in rows) / sum(r["labelled"] for r in rows)))
print("Spearman(proxy, accuracy) = %+.3f   (reference +0.82 with best_v2.pt)" % rho)
print("%d of the %d image(s) with <= 6 names right are among the 10 worst by proxy" % (
    len(set(low) & worst10), len(low)))
print("VERDICT: %s" % ("the proxy ranks pages well enough to pick what to audit" if rho >= 0.7 else
                       "the proxy does NOT follow accuracy with this model - do not rank with it; audit a "
                       "larger random sample instead and tell the user"))
if args.out:
    write_csv(args.out, rows)
