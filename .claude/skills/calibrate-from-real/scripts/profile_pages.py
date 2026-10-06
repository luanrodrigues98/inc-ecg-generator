#!/usr/bin/env python
"""Step 3 - a measured profile of each page, by the SAME code on real and synthetic pages.

    real lot:   profile_pages.py <lot_dir> <out.csv> --run-dir <run_dir>
    synthetic:  profile_pages.py <synthetic_dir> <out.csv> --synthetic [--sample 25]

Nothing here uses ground truth, which is what lets one code path serve both sides; the
lead detector being calibrated is not used either (measuring structure with the model one
wants to improve is circular). A synthetic page additionally gets truth_* columns copied
from its JSON sidecar, to split the set by population and to check the measures themselves.

Per page:
  px_per_mm, scale_source   the grid scale: from the 5 mm lattice the digitizer's rectifier finds
                            (lattice_gated when its gate approves the lattice, lattice_ungated
                            otherwise), else the 1 mm period of rectifier.scale.bootstrap_scale
                            (bootstrap), else empty - and every mm value below is empty too
  scale_suspect             1 when the scale implies a frame whose long side is outside
                            180-450 mm: no ECG sheet is that size, so the estimate is probably
                            a harmonic of the grid (common on low-resolution photos). Leave
                            suspect pages out of every statistic in mm
  skew_deg, gate_pass, gate_reason   the rectifier's own reading of the page
  page_w_mm, page_h_mm      physical size of the frame (the sheet, on a scan; sheet +
                            background, on a photo)
  paper_r/g/b, paper_lum    the paper between the grid rules (bright pixels)
  tone_r/g/b                paper and grid together, traces left out (what the page "reads as")
  grid_cover                share of non-trace pixels that are chromatic: 0 on a grey grid
  grid_r/g/b, grid_hue_deg  the chromatic (grid) pixels
  colourfulness             mean chroma of the non-trace pixels: ~0 on a black-and-white copy
  paper_noise               robust high-frequency noise on the paper, in grey levels
  trace_fwhm_mm, trace_core width at half depth and darkest level of the dark strokes, on the
                            max(R,G,B) plane - a coloured grid rule stays bright in its own
                            channel there, a trace is dark in all three (the Cardio numbers in
                            the batch YAML were taken on R, which is the same thing on pink paper)

  alias_1440, alias_1024   moire a plain linear shrink to the detector's input creates
                            (_common.aliasing_score). Generator pages alias too: compare
                            at matched resolution (--synth-filter on truth_dpi)

Lead-name position is NOT measured here. A pixel-profile measure (labels.pt text boxes
against the nearest row of dark pixels) was tried and agreed with the JSON truth on only
5 of 7 synthetic pages, finding 3 names or more on fewer than half of them: the grid rules
out-vote a thin trace. Take the name position from the visual audit (step 2) and, on the
synthetic side, from truth_name_position.
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_widths

from _common import (DIGITIZER, aliasing_score, list_images, read_image, require_orientation, slug, upright,
                     usable_rows, use_digitizer_src, write_csv)

parser = argparse.ArgumentParser()
parser.add_argument("image_dir")
parser.add_argument("out_csv")
parser.add_argument("--run-dir", help="real lot: where manifest.csv and orientation.csv are")
parser.add_argument("--synthetic", action="store_true", help="generator output: upright pages with JSON sidecars")
parser.add_argument("--sample", type=int, default=0, help="profile a random sample of this many pages")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--scale", choices=["rectify", "bootstrap"], default="rectify",
                    help="rectify (default) runs the rectifier network, ~10 s a page on CPU; bootstrap is "
                         "instant but fails or lands on a harmonic on roughly a third of the pages")
args = parser.parse_args()
if bool(args.run_dir) == bool(args.synthetic):
    raise SystemExit("give exactly one of --run-dir (real lot) or --synthetic")

use_digitizer_src()
from rectifier.scale import bootstrap_scale  # noqa: E402

rect_model = rect_cfg = None
if args.scale == "rectify":
    from rectifier.rectify import load_for_inference, rectify  # noqa: E402
    rect_model, rect_cfg = load_for_inference(DIGITIZER / "models" / "rectifier_400dpi.pth", "cpu")


def scale(page, out):
    """px per mm of the page, or None."""
    h, w = page.shape[:2]
    px_mm, source = None, ""
    out.update(skew_deg="", gate_pass="", gate_reason="")
    if rect_model is not None:
        result = rectify(page, rect_model, rect_cfg, device="cpu", warp=False)
        out.update(gate_pass=int(result.gate_pass), gate_reason=result.gate_reason,
                   skew_deg="" if result.skew_deg is None else round(float(result.skew_deg), 2))
        if result.px_per_mm and np.isfinite(result.px_per_mm):
            px_mm, source = float(result.px_per_mm), "lattice_gated" if result.gate_pass else "lattice_ungated"
    if px_mm is None:
        estimate = bootstrap_scale(page)
        if estimate.ok:
            px_mm, source = float(estimate.px_per_mm), "bootstrap"
    out.update(scale_source=source, px_per_mm=round(px_mm, 3) if px_mm else "",
               dpi=round(px_mm * 25.4) if px_mm else "",
               page_w_mm=round(w / px_mm, 1) if px_mm else "", page_h_mm=round(h / px_mm, 1) if px_mm else "",
               scale_suspect=int(not 180 <= max(h, w) / px_mm <= 450) if px_mm else "")
    return px_mm


def colour(page, out):
    h, w = page.shape[:2]
    crop = page[int(0.25 * h):int(0.75 * h):2, int(0.25 * w):int(0.75 * w):2, ::-1].astype(np.int16)  # RGB
    chroma = crop.max(2) - crop.min(2)
    lum = crop.mean(2)
    not_trace = ~((chroma < 40) & (lum < 180))
    grid = not_trace & (chroma > 30)
    paper = not_trace & (lum >= np.percentile(lum[not_trace], 70)) if not_trace.any() else not_trace
    out["grid_cover"] = round(float(grid.sum() / max(not_trace.sum(), 1)), 4)
    out["colourfulness"] = round(float(chroma[not_trace].mean()), 2) if not_trace.any() else ""
    for key, mask in (("tone", not_trace), ("paper", paper), ("grid", grid)):
        rgb = np.median(crop[mask], axis=0) if mask.sum() > 50 else [np.nan] * 3
        for channel, value in zip("rgb", rgb):
            out["%s_%s" % (key, channel)] = "" if np.isnan(value) else int(value)
    out["paper_lum"] = round(float(np.median(lum[paper])), 1) if paper.any() else ""
    if grid.sum() > 50:
        hsv = cv2.cvtColor(crop[grid].astype(np.uint8)[None], cv2.COLOR_RGB2HSV)[0]
        angle = hsv[:, 0].astype(float) * 2 * np.pi / 180  # OpenCV hue is degrees / 2
        out["grid_hue_deg"] = round(float(np.degrees(np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())) % 360), 1)
    else:
        out["grid_hue_deg"] = ""
    grey = lum.astype(np.float32)
    residual = np.abs(grey - cv2.medianBlur(grey, 3))[paper]
    out["paper_noise"] = round(float(1.4826 * np.median(residual)), 2) if residual.size else ""


def trace(page, px_mm, out):
    h, w = page.shape[:2]
    value = page[int(0.25 * h):int(0.75 * h), int(0.25 * w):int(0.75 * w)].max(2).astype(float)
    paper = np.percentile(value, 60)
    scale = px_mm if px_mm else 12.0
    widths, cores = [], []
    for x in range(0, value.shape[1], 7):
        dip = paper - value[:, x]
        peaks, _ = find_peaks(dip, prominence=35, distance=max(3, int(0.8 * scale)))
        if not len(peaks):
            continue
        width = peak_widths(dip, peaks, rel_height=0.5)[0]
        keep = width < 1.0 * scale
        widths.extend(width[keep])
        cores.extend(value[peaks[keep], x])
    out["trace_fwhm_px"] = round(float(np.median(widths)), 2) if widths else ""
    out["trace_fwhm_mm"] = round(float(np.median(widths)) / px_mm, 3) if widths and px_mm else ""
    out["trace_core"] = int(np.median(cores)) if cores else ""
    out["trace_strokes"] = len(widths)


def row_pitch(data):
    """Median baseline-to-baseline distance (mm) of the lead rows of the first column."""
    xs, ys = [], []
    for lead in data.get("leads", []):
        pts = lead.get("plotted_pixels")
        if not pts:
            continue
        arr = np.asarray(pts, float)
        arr = arr[np.isfinite(arr).all(1)]
        if len(arr) < 5:
            continue
        xs.append(arr[:, 0].min())
        ys.append(float(np.median(arr[:, 1])))
    if len(ys) < 3:
        return ""
    xs, ys = np.array(xs), np.array(ys)
    # the first column: traces starting within 10% of the page width of the leftmost one
    # (columns sit ~62 mm apart on a 3x4; a skewed 12x1 spreads its starts by a few mm)
    first = np.sort(ys[xs <= xs.min() + 0.10 * float(data.get("width") or (xs.max() + 1))])
    steps = np.diff(first)
    return round(float(np.median(steps)) / (data["resolution"] / 25.4), 1) if len(steps) else ""


# ecg_plot.standard_major_colors x 255: the palette index of a synthetic page, from its JSON
PALETTE_MAJOR = {1: (109, 50, 47), 2: (255, 203, 221), 3: (0, 0, 102), 4: (0, 77, 0), 5: (255, 0, 0),
                 6: (102, 102, 102), 7: (248, 130, 160), 8: (252, 156, 112), 9: (246, 184, 166)}


def palette_of(major):
    if not major:
        return ""
    best = min(PALETTE_MAJOR, key=lambda k: sum((a - b) ** 2 for a, b in zip(PALETTE_MAJOR[k], major)))
    return best if sum((a - b) ** 2 for a, b in zip(PALETTE_MAJOR[best], major)) < 75 else ""


def truth(path, out):
    sidecar = path.with_suffix(".json")
    if not sidecar.exists():
        return
    try:
        data = json.loads(sidecar.read_text())
    except ValueError:
        return  # an interrupted batch can leave a truncated sidecar behind
    out["truth_row_pitch_mm"] = row_pitch(data)
    out["truth_palette"] = palette_of(data.get("grid_line_color_major"))
    out["truth_columns"] = data.get("number_of_columns_in_image", "")
    out["truth_name_position"] = data.get("lead_name_position", "")
    out["truth_dpi"] = data.get("resolution", "")
    out["truth_trace_mm"] = data.get("trace_thickness_mm", "")
    paper = data.get("paper_color")
    out["truth_paper"] = "" if paper is None else "/".join(str(int(v)) for v in paper)
    for group, flag in (data.get("realism") or {}).items():
        out["truth_" + group] = int(bool(flag))


if args.synthetic:
    paths = [p for p in list_images(args.image_dir) if p.with_suffix(".json").exists()]
    broken = [p for p in paths if p.with_suffix(".json").stat().st_size == 0]
    if broken:
        print("skipping %d page(s) with an empty JSON sidecar (interrupted batch): %s" % (
            len(broken), ", ".join(p.name for p in broken[:5])))
        paths = [p for p in paths if p not in broken]
    todo = [(p, slug(p, args.image_dir), 0) for p in paths]
else:
    orientation = require_orientation(args.run_dir)
    todo = [(Path(args.image_dir) / r["file"], r["slug"], orientation[r["file"]]) for r in usable_rows(args.run_dir)]
if args.sample and len(todo) > args.sample:
    todo = sorted(random.Random(args.seed).sample(todo, args.sample), key=lambda t: t[1])

rows = []
for path, name, rot in todo:
    bgr, _status = read_image(path)
    if bgr is None:
        continue
    page = upright(bgr, rot)
    h, w = page.shape[:2]
    out = dict(slug=name, width=w, height=h, aspect=round(w / h, 4))
    px_mm = scale(page, out)
    if out["scale_suspect"] == 1:
        px_mm = None  # a harmonic would put every mm measure off by its factor
    colour(page, out)
    trace(page, px_mm, out)
    out["alias_1440"] = aliasing_score(page, 1440)
    out["alias_1024"] = aliasing_score(page, 1024)
    if args.synthetic:
        truth(path, out)
    rows.append(out)
    print("%-28s %s px/mm (%s)  paper %s/%s/%s  cover %s  fwhm %s mm" % (
        name[:28], out["px_per_mm"] or "?", out["scale_source"] + ("?" if out["scale_suspect"] == 1 else ""), out["paper_r"], out["paper_g"], out["paper_b"], out["grid_cover"],
        out["trace_fwhm_mm"] or "?"), flush=True)

columns = []
for row in rows:
    columns += [key for key in row if key not in columns]
write_csv(args.out_csv, rows, columns)
usable = sum(bool(r["px_per_mm"]) and r["scale_suspect"] == 0 for r in rows)
print("wrote %s: %d page(s), %d with a usable grid scale, %d suspect (harmonic?), %d without" % (
    args.out_csv, len(rows), usable, sum(r["scale_suspect"] == 1 for r in rows),
    sum(not r["px_per_mm"] for r in rows)))
