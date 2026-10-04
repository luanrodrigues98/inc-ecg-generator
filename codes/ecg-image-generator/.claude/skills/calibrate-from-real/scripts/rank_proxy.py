#!/usr/bin/env python
"""Step 1 - rank the pages of a lot by how well the current YOLO copes, without labels.

    rank_proxy.py <lot_dir> <run_dir> [--model models/best_v2.pt]

Needs <run_dir>/manifest.csv and orientation.csv (intake.py). Writes <run_dir>/ranking.csv,
worst page first: proxy = mean confidence of the leads the pipeline keeps x (leads / 12).
The model path is relative to the digitizer unless absolute. CPU, about 1-2 s per page.

Also scores each page for moire at the detector's input size (alias_detector) and at
1024 px (alias_1024, the imgsz nb 4.3 trains new models at): moire appears or vanishes with
the input size, so a page can look clean to the current model and not to the next one.

The proxy ranks pages; it does not explain them. The worst pages are where to LOOK
(audit), not a list of gaps by themselves - a page can score low for a reason the
generator already covers, and a gap can hide on a page that scores well.
"""
import argparse
from pathlib import Path

import numpy as np

from _common import (DIGITIZER, LEAD_NAMES, aliasing_score, detect_leads, load_yolo, proxy_score, read_image,
                     require_orientation, upright, usable_rows, write_csv)

parser = argparse.ArgumentParser()
parser.add_argument("lot_dir")
parser.add_argument("run_dir")
parser.add_argument("--model", default="models/best_v2.pt")
args = parser.parse_args()

model_path = Path(args.model) if Path(args.model).is_absolute() else DIGITIZER / args.model
model = load_yolo(model_path)
imgsz = int(model.overrides.get("imgsz") or 640)
orientation = require_orientation(args.run_dir)
names = [n.lower() for n in LEAD_NAMES]

rows = []
for row in usable_rows(args.run_dir):
    bgr, _status = read_image(Path(args.lot_dir) / row["file"])
    page = upright(bgr, orientation[row["file"]])
    leads, result = detect_leads(model, page)
    confs = [c for _q, c in leads.values()]
    raw = result.obb.conf.cpu().numpy() if result.obb is not None else result.boxes.conf.cpu().numpy()
    rows.append(dict(idx=row["idx"], file=row["file"], slug=row["slug"],
                     proxy=round(proxy_score(leads), 4), n_leads=len(leads),
                     mean_conf=round(float(np.mean(confs)), 4) if confs else "",
                     n_raw=len(raw), missing=" ".join(n for n in names if n not in leads),
                     alias_detector=aliasing_score(page, imgsz), alias_1024=aliasing_score(page, 1024)))
    print("%4s proxy %.3f leads %2d  %s" % (row["idx"], rows[-1]["proxy"], len(leads), row["file"]), flush=True)

rows.sort(key=lambda r: r["proxy"])
for rank, r in enumerate(rows):
    r["rank"] = rank
write_csv(Path(args.run_dir) / "ranking.csv", rows,
          ["rank", "idx", "file", "slug", "proxy", "n_leads", "mean_conf", "n_raw", "missing", "alias_detector",
           "alias_1024"])

proxy = np.array([r["proxy"] for r in rows])
n12 = np.mean([r["n_leads"] == 12 for r in rows])
print("\nmodel %s, %d page(s)" % (model_path.name, len(rows)))
print("proxy p10 %.3f  median %.3f  p90 %.3f | leads per page %.1f | pages with 12 leads %.0f%%" % (
    np.percentile(proxy, 10), np.median(proxy), np.percentile(proxy, 90),
    np.mean([r["n_leads"] for r in rows]), 100 * n12))
print("reference with best_v2.pt on the 104 Cardio scans: 11.4 leads per page on average, 69% with 12")
print("worst pages:")
for r in rows[:10]:
    print("  #%-4s proxy %.3f leads %2d  %s" % (r["idx"], r["proxy"], r["n_leads"], r["file"]))
cut = max(1.0, float(np.percentile([r["alias_1024"] for r in rows], 80)))
flagged = [r for r in sorted(rows, key=lambda r: -r["alias_1024"]) if r["alias_1024"] >= cut]
print("moire candidates (alias_1024 >= %.2f: 1.0 or the lot's p80, whichever is higher - the audit shows them at "
      "1024 px too): %s" % (cut, ", ".join("#%s %.1f/%.1f" % (r["idx"], r["alias_detector"], r["alias_1024"])
                                            for r in flagged) or "none"))
print("NB generator pages alias as well: compare at MATCHED resolution before calling moire a gap "
      "(compare_profiles.py alias rows with --synth-filter \"truth_dpi < 420\" or whatever matches the lot)")
print("wrote %s/ranking.csv" % args.run_dir)
