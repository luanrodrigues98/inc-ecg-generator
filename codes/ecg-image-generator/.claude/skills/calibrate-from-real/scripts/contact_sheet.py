#!/usr/bin/env python
"""Step 5 - real pages beside synthetic ones, to LOOK at what the numbers cannot say.

    contact_sheet.py <lot_dir> <run_dir> <synthetic_dir> <out_LOCAL.png>
                     [--n 6] [--population NAME --populations populations.csv] [--slugs a,b,c]

Left column real, right column synthetic, two rows per pair:
    whole page, 900 px wide                     layout, name placement, page furniture
    a 70 x 45 mm window at 12 px/mm             grid, trace and printed names at one
                                                physical scale on both sides
The window is the one in the left half of the page, below the header band and above the
footer, with the most dark ink (where traces start and names are printed), chosen by the
same rule on both sides. A real page whose
grid scale is missing or suspect (profile_real.csv in <run_dir>) gets a same-size window
at its native pixels instead, marked "native px".

Real pages: --slugs, else the given population, else an even spread over the proxy
ranking (best to worst). Synthetic pages: an even spread over the directory, by name.

The output name must end in _LOCAL.png: real pages carry patient names, and this sheet
is for looking at here, not for publishing.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from _common import list_images, read_csv, read_image, require_orientation, upright

parser = argparse.ArgumentParser()
parser.add_argument("lot_dir")
parser.add_argument("run_dir")
parser.add_argument("synthetic_dir")
parser.add_argument("out_png")
parser.add_argument("--n", type=int, default=6)
parser.add_argument("--populations")
parser.add_argument("--population")
parser.add_argument("--slugs")
args = parser.parse_args()
if not args.out_png.endswith("_LOCAL.png"):
    raise SystemExit("the output name must end in _LOCAL.png (it shows real patient pages)")

PAGE_W, MM, WIN = 900, 12, (70, 45)
run_dir = Path(args.run_dir)
orientation = require_orientation(run_dir)
ranking = {row["slug"]: row for row in read_csv(run_dir / "ranking.csv")}
scales = {}
if (run_dir / "profile_real.csv").exists():
    for row in read_csv(run_dir / "profile_real.csv"):
        if row.get("px_per_mm") and row.get("scale_suspect") in ("0", "0.0"):
            scales[row["slug"]] = float(row["px_per_mm"])


def spread(items, n):
    if len(items) <= n:
        return list(items)
    return [items[int(round(k))] for k in np.linspace(0, len(items) - 1, n)]


if args.slugs:
    real = [ranking[s] for s in args.slugs.split(",")]
elif args.population:
    members = {r["slug"] for r in read_csv(args.populations) if r["population"] == args.population}
    real = spread([r for r in ranking.values() if r["slug"] in members][::-1], args.n)
else:
    real = spread(list(ranking.values())[::-1], args.n)
synthetic = spread([p for p in list_images(args.synthetic_dir) if p.with_suffix(".json").exists()], len(real))
if not real or not synthetic:
    raise SystemExit("nothing to show: %d real, %d synthetic page(s)" % (len(real), len(synthetic)))


def caption(text, width, colour=(0, 0, 0)):
    bar = np.full((30, width, 3), 255, np.uint8)
    cv2.putText(bar, text[:int(width / 9)], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return bar


def busiest_window(page, ww, wh):
    """Top-left corner of the ww x wh window in the left half of the page holding the most
    dark ink: where a trace starts and its name is printed. The same rule on both sides."""
    value = page.max(2)
    # near-black is not ink: rotation corners, sensor marks, scanner lids
    dark = ((value < np.percentile(value[::8, ::8], 60) - 60) & (value > 30)).astype(np.float32)
    total = cv2.integral(dark)
    h, w = dark.shape
    best, corner = -1.0, (0, 0)
    # stay off the header band (pen notes, stamps) and the footer
    y_from, y_to = int(0.15 * h), max(int(0.90 * h) - wh, int(0.15 * h))
    for y in range(y_from, y_to + 1, max(wh // 4, 1)):
        for x in range(0, max(w // 2 - ww // 2, 0) + 1, max(ww // 4, 1)):
            ink = total[y + wh, x + ww] - total[y, x + ww] - total[y + wh, x] + total[y, x]
            if ink > best:
                best, corner = ink, (x, y)
    return corner


def tile(page, px_mm, text, colour):
    h, w = page.shape[:2]
    whole = cv2.resize(page, (PAGE_W, int(round(h * PAGE_W / w))), interpolation=cv2.INTER_AREA)
    ww, wh = (int(WIN[0] * px_mm), int(WIN[1] * px_mm)) if px_mm else (WIN[0] * MM, WIN[1] * MM)
    ww, wh = min(ww, w), min(wh, h)
    x0, y0 = busiest_window(page, ww, wh)
    window = page[y0:y0 + wh, x0:x0 + ww]
    note = "70 x 45 mm at 12 px/mm" if px_mm else "native px (no usable grid scale)"
    interpolation = cv2.INTER_AREA if window.shape[1] > WIN[0] * MM else cv2.INTER_CUBIC
    window = cv2.resize(np.ascontiguousarray(window), (WIN[0] * MM, WIN[1] * MM), interpolation=interpolation)
    window = np.pad(window, ((0, 0), (0, PAGE_W - window.shape[1]), (0, 0)), constant_values=255)
    return np.vstack([caption(text, PAGE_W, colour), whole, caption(note, PAGE_W, (110, 110, 110)), window])


pairs = []
for row, synth_path in zip(real, synthetic):
    bgr, _status = read_image(Path(args.lot_dir) / row["file"])
    left = tile(upright(bgr, orientation[row["file"]]), scales.get(row["slug"]),
                "REAL %s  proxy %s  leads %s" % (row["slug"], row["proxy"], row["n_leads"]), (0, 0, 170))
    data = json.loads(synth_path.with_suffix(".json").read_text())
    flags = "+".join(k for k, v in (data.get("realism") or {}).items() if v)
    right = tile(cv2.imread(str(synth_path)), data["resolution"] / 25.4,
                 "SYNTHETIC %s  %s col  name %s  %s" % (synth_path.stem, data.get("number_of_columns_in_image"),
                                                         data.get("lead_name_position"), flags), (0, 110, 0))
    height = max(left.shape[0], right.shape[0])
    left, right = [np.pad(t, ((0, height - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in (left, right)]
    pairs.append(np.hstack([left, np.full((height, 24, 3), 225, np.uint8), right]))
sheet = np.vstack([np.pad(p, ((0, 24), (0, 0), (0, 0)), constant_values=225) for p in pairs])
cv2.imwrite(args.out_png, sheet)
print("wrote %s (%d pair(s), %d x %d px) - local only" % (args.out_png, len(pairs), sheet.shape[1], sheet.shape[0]))
