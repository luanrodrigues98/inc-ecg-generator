#!/usr/bin/env python
"""Step 2 - pick the pages to audit and cut the views to look at.

    audit_views.py <lot_dir> <run_dir> [--worst 12] [--control 12 | --all] [--seed 0] [--width 1600]
                   [--model models/best_v2.pt]

Selection: the `worst` pages by proxy (where gaps are DISCOVERED) and a random `control`
sample of the rest (where frequencies are COUNTED - the worst pages are a biased sample by
construction). --all, or a lot no larger than worst + control, audits the whole lot, every
page as control; the `rank` column still says which pages are the worst by proxy.

For each selected page, in <run_dir>/audit/<slug>/, all upright:
    audit.jpg    ONE composite: the detector view, then the row ends, detail, header, footer -
                 read this one first; one image a page keeps an audit affordable
    detections.csv  every raw box (name, confidence, kept by the pipeline or not, centre and
                 size in page pixels) - countable evidence of boxes crossing rows or sitting
                 on a calibration pulse
    detector.jpg the page AS THE DETECTOR RECEIVES IT: shrunk to the model's input size with
                 plain linear interpolation (what Ultralytics does - no antialiasing, so a
                 fine grid aliases into moire here and nowhere else), with the leads the
                 pipeline keeps in green (name, confidence) and the discarded boxes in red
    at1024.jpg   moire candidates only (as rank_proxy.py flags them): the page at 1024 px, linear - the
                 size nb 4.3 trains new models at; moire comes and goes with the input size
    page.jpg     whole page, long side 1600, clean  layout, panels, header/footer, handwriting
    ends.jpg     the two ends of the lead rows      calibration pulse at the left or the RIGHT
                 (left 30% and right 25% of the top  end, lead names, device text next to them
                 half, side by side)
    topleft.jpg  left 45% x top 50%, <= 1800 px  lead names: vocabulary and position
    detail.jpg   900 x 900 px at NATIVE scale    grid style, trace width, moire, JPEG blocks
    header.jpg   top 14% strip                   printer format, patient block
    footer.jpg   bottom 14% strip                speed/gain line, date-time stamp (exam identity)
Open a single view only when the composite leaves a field undecided (small print in a
footer, a grid too fine to judge). Scans run to 5000 x 6500 px: never read the originals.

Also writes <run_dir>/audit/selection.csv. The views show patient names: local only.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from _common import (DIGITIZER, aliasing_score, detect_leads, load_yolo, read_csv, read_image,
                     require_orientation, shrink, upright, write_csv)

WIDTH = 1600  # --width; every page of the audit is one image read, so this sets its cost


def fit(view, width, max_height=None):
    scale = width / view.shape[1]
    if max_height:
        scale = min(scale, max_height / view.shape[0])
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(view, None, fx=scale, fy=scale, interpolation=interpolation)


def labelled(view, text, width):
    view = np.pad(view, ((0, 0), (0, width - view.shape[1]), (0, 0)), constant_values=255)
    bar = np.full((26, width, 3), 235, np.uint8)
    cv2.putText(bar, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return np.vstack([bar, view])


def tag(view, pts, text, colour, bottom=False):
    """A readable label on a white patch: kept boxes at their top-left corner, discarded
    ones at their bottom-left, so the two do not cover each other."""
    x, y = int(pts[:, 0].min()), int(pts[:, 1].max() if bottom else pts[:, 1].min())
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y = max(th + 4, min(y if bottom else y + th + 4, view.shape[0] - 2))
    cv2.rectangle(view, (x, y - th - 4), (x + tw + 4, y + 2), (255, 255, 255), -1)
    cv2.putText(view, text, (x + 2, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)


def detector_view(page, model):
    """The page at the model's input size, as Ultralytics resizes it, with the boxes drawn."""
    imgsz = int(model.overrides.get("imgsz") or 640)
    leads, result = detect_leads(model, page)
    h, w = page.shape[:2]
    scale = imgsz / max(h, w)
    view = cv2.resize(page, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
    if result.obb is not None:
        corners = result.obb.xyxyxyxy.cpu().numpy()
        names = [model.names[int(c)] for c in result.obb.cls.cpu().numpy()]
        confs = result.obb.conf.cpu().numpy()
    else:
        xyxy = result.boxes.xyxy.cpu().numpy()
        corners = np.stack([xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [2, 3]], xyxy[:, [0, 3]]], axis=1)
        names = [model.names[int(c)] for c in result.boxes.cls.cpu().numpy()]
        confs = result.boxes.conf.cpu().numpy()
    kept = [quad for quad, _conf in leads.values()]
    for quad, name, conf in zip(corners, names, confs):
        if any(np.allclose(quad, k, atol=1.0) for k in kept):
            continue
        pts = (quad * scale).astype(np.int32)
        cv2.polylines(view, [pts], True, (40, 40, 220), 1, cv2.LINE_AA)
        tag(view, pts, "%s %.2f" % (name, conf), (40, 40, 200), bottom=True)
    for name, (quad, conf) in leads.items():
        pts = (quad * scale).astype(np.int32)
        cv2.polylines(view, [pts], True, (30, 150, 30), 2, cv2.LINE_AA)
        tag(view, pts, "%s %.2f" % (name, conf), (20, 110, 20))
    rows = []
    for quad, name, conf in zip(corners, names, confs):
        kept_box = any(np.allclose(quad, k, atol=1.0) for k in kept)
        rows.append(dict(name=name, conf=round(float(conf), 3), kept=int(kept_box),
                         cx=round(float(quad[:, 0].mean()), 1), cy=round(float(quad[:, 1].mean()), 1),
                         w=round(float(np.ptp(quad[:, 0])), 1), h=round(float(np.ptp(quad[:, 1])), 1)))
    return view, imgsz, rows


def composite(views, imgsz):
    """detector view on top; the page at 1024 px when it is a moire candidate; the row ends
    beside the native-scale detail; header and footer."""
    left, right = int(WIDTH * 0.58), WIDTH - int(WIDTH * 0.58) - 8
    names = labelled(fit(views["ends"], left, 760), "row ends: left 30% | right 25% (pulse, names, device text)", left)
    detail = labelled(fit(views["detail"], right, 760), "detail: native pixels (shown at up to %d px)" % right, right)
    height = max(names.shape[0], detail.shape[0])
    names, detail = [np.pad(v, ((0, height - v.shape[0]), (0, 0), (0, 0)), constant_values=255) for v in (names, detail)]
    middle = np.hstack([names, np.full((height, 8, 3), 255, np.uint8), detail])
    top = views["detector"]
    scale = min(WIDTH / top.shape[1], 1000 / top.shape[0])
    # nearest neighbour: the detector's pixels are what matters here, not a smooth picture
    top = cv2.resize(top, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST if scale > 1 else cv2.INTER_AREA)
    caption = ("as the detector sees it (%d px, linear, no antialias): green kept by the pipeline, red discarded"
               % imgsz)
    extra = []
    if "at1024" in views:
        a = views["at1024"]
        k = min(WIDTH / a.shape[1], 700 / a.shape[0])
        a = cv2.resize(a, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST if k > 1 else cv2.INTER_AREA)
        extra = [labelled(a, "moire candidate: the page at 1024 px, linear (nb 4.3's training size) - look for beat bands",
                          WIDTH)]
    return np.vstack([labelled(top, caption, WIDTH)] + extra + [middle,
                      labelled(fit(views["header"], WIDTH, 170), "header", WIDTH),
                      labelled(fit(views["footer"], WIDTH, 170), "footer", WIDTH)])


parser = argparse.ArgumentParser()
parser.add_argument("lot_dir")
parser.add_argument("run_dir")
parser.add_argument("--worst", type=int, default=12)
parser.add_argument("--control", type=int, default=12)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--model", default="models/best_v2.pt")
parser.add_argument("--all", action="store_true", help="audit the whole lot (every page a control page)")
parser.add_argument("--width", type=int, default=1600)
args = parser.parse_args()
WIDTH = args.width
model = load_yolo(args.model if args.model.startswith("/") else DIGITIZER / args.model)

run_dir = Path(args.run_dir)
orientation = require_orientation(run_dir)
ranking = read_csv(run_dir / "ranking.csv")  # worst first
# the moire candidates rank_proxy.py flags: 1.0 or the lot's p80 of alias_1024, whichever is higher
_alias = sorted(float(r["alias_1024"]) for r in ranking if r.get("alias_1024") not in (None, ""))
alias_cut = max(1.0, _alias[int(0.8 * (len(_alias) - 1))]) if _alias else float("inf")
if args.all or len(ranking) <= args.worst + args.control:
    chosen = [(row, "control") for row in ranking]
else:
    rest = ranking[args.worst:]
    control = random.Random(args.seed).sample(rest, args.control)
    chosen = [(row, "worst") for row in ranking[:args.worst]] + [(row, "control") for row in control]

selection = []
for row, group in chosen:
    bgr, _status = read_image(Path(args.lot_dir) / row["file"])
    page = upright(bgr, orientation[row["file"]])
    h, w = page.shape[:2]
    out = run_dir / "audit" / row["slug"]
    out.mkdir(parents=True, exist_ok=True)
    side = min(900, h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    views = {
        "page": shrink(page, 1600),
        "topleft": shrink(page[:h // 2, :int(0.45 * w)], 1800),
        "ends": shrink(np.hstack([page[:h // 2, :int(0.30 * w)], np.full((h // 2, max(8, w // 200), 3), 255, np.uint8),
                                  page[:h // 2, int(0.75 * w):]]), 1800),
        "detail": page[y0:y0 + side, x0:x0 + side],
        "header": shrink(page[:int(0.14 * h)], 1800),
        "footer": shrink(page[int(0.86 * h):], 1800),
    }
    views["detector"], imgsz, boxes = detector_view(page, model)
    if row.get("alias_1024") not in (None, "") and float(row["alias_1024"]) >= alias_cut:
        scale = 1024 / max(h, w)
        views["at1024"] = cv2.resize(page, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
    if boxes:  # every raw box, kept or not, in page pixels: rows crossed, pulses taken for leads
        write_csv(out / "detections.csv", boxes)
    for name, view in views.items():
        cv2.imwrite(str(out / (name + ".jpg")), view, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out / "audit.jpg"), composite(views, imgsz), [cv2.IMWRITE_JPEG_QUALITY, 88])
    selection.append(dict(idx=row["idx"], file=row["file"], slug=row["slug"], group=group, rank=row["rank"],
                          proxy=row["proxy"], n_leads=row["n_leads"], missing=row["missing"],
                          width=w, height=h, views=str(out)))
    print("%-7s #%-4s proxy %s  %s" % (group, row["idx"], row["proxy"], out), flush=True)

write_csv(run_dir / "audit" / "selection.csv", selection)
print("\n%d page(s) to audit: %d in the worst-by-proxy group, %d control (%d of them among the %d lowest ranks). "
      "Record one JSON line per page in %s/audit.jsonl (fields and vocabulary: references/audit-checklist.md)" % (
          len(selection), sum(s["group"] == "worst" for s in selection),
          sum(s["group"] == "control" for s in selection),
          sum(s["group"] == "control" and int(s["rank"]) < args.worst for s in selection), args.worst, run_dir))
