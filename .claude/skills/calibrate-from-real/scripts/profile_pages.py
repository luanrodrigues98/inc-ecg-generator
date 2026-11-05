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

  grid5_fwhm_px400, grid1_fwhm_px400, grid5_contrast, grid1_contrast, grid_tiles, page400_source
                            width (FWHM, px) and contrast (L* x 100/255, paper minus rule) of
                            the horizontal 5 mm and 1 mm grid rules, on the 400 dpi page the
                            digitizer itself builds (digitizer.prepare_page with the pipeline's
                            rectifier_v2.pth) - never in native pixels, where one pixel of a
                            90 dpi page is already ~0.28 mm. The method of the per-lead
                            reference (~/.cache/inc-ecg-digitizer-run/colnet-linha-grade/
                            fwhm_local.py) applied to short tiles of the whole page: per tile,
                            the row profile of L = median over the tile's columns, with the
                            dark strokes (trace, text) and 10 px above and below them left out;
                            the 1 mm and 5 mm phases of that profile; per rule, the minimum
                            within +-3 px of its phase, the background = the maximum within
                            +-7 px, the FWHM = contiguous rows below half-way, kept when the
                            contrast is >= 4. Median over the rules of a tile, then over the
                            tiles. Tiles are short (GRID_TILE_W px) because a row median turns
                            skew into width: 0.2 deg over 4500 px moves a rule ~16 px. Empty
                            when the page could not be brought to 400 dpi (page400_source
                            "original") or no tile shows a periodic grid. --no-grid skips it
                            (it runs a second rectifier: +5-25 s a page on CPU)

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
parser.add_argument("--no-grid", action="store_true",
                    help="skip the grid rule widths (grid5_*/grid1_*): they bring the page to 400 dpi with the "
                         "digitizer's prepare_page, a second rectifier pass of 5-25 s a page on CPU")
args = parser.parse_args()
if bool(args.run_dir) == bool(args.synthetic):
    raise SystemExit("give exactly one of --run-dir (real lot) or --synthetic")

use_digitizer_src()
from rectifier.scale import bootstrap_scale  # noqa: E402

rect_model = rect_cfg = None
if args.scale == "rectify":
    from rectifier.rectify import load_for_inference, rectify  # noqa: E402
    rect_model, rect_cfg = load_for_inference(DIGITIZER / "models" / "rectifier_400dpi.pth", "cpu")

page_rectifier = None
if not args.no_grid:
    # The pipeline's own page: the app loads rectifier_v2.pth (app/app.py), not the
    # rectifier_400dpi.pth the scale columns above have always used - those stay as they were,
    # so older profile CSVs remain comparable.
    from digitizer.digitizer import prepare_page  # noqa: E402
    from rectifier.rectify import load_for_inference as _load_rectifier  # noqa: E402
    page_rectifier = _load_rectifier(DIGITIZER / "models" / "rectifier_v2.pth", "cpu")


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


# Grid rule widths on the 400 dpi page (see the docstring). Tile sizes in px of that page.
GRID_TILE_W, GRID_TILE_H = 240, 400
GRID_MIN_PERIODICITY = 1.0   # L darker on the 1 mm rows than half a period off: below it, no grid
GRID_COLUMNS = ["grid5_fwhm_px400", "grid1_fwhm_px400", "grid5_contrast", "grid1_contrast", "grid_tiles"]


def grid_phase(lrow, period):
    """(f1, f5, periodicity) of a row profile of L: the 1 mm phase, the 5 mm phase among the
    five 1 mm ones, and how much darker the 1 mm rows are than the rows half a period off."""
    h = len(lrow)
    ink = 100.0 - lrow

    def mean_at(phase, step):
        rows = phase + step * np.arange(-1, h / step + 1)
        rows = rows[(rows >= 0) & (rows <= h - 1)]
        return np.interp(rows, np.arange(h), ink).mean() if len(rows) else 0.0

    f1 = max(np.arange(0, period, 0.25), key=lambda f: mean_at(f, period))
    f5 = max((f1 + k * period for k in range(5)), key=lambda f: mean_at(f, 5 * period))
    return f1, f5, mean_at(f1, period) - mean_at(f1 + period / 2, period)


def rule_fwhm(lrow, row):
    """(fwhm px, contrast) of the rule nearest `row` in a row profile, or None below contrast 4."""
    h = len(lrow)
    a0, a1 = max(0, row - 3), min(h, row + 4)
    row = a0 + int(np.argmin(lrow[a0:a1]))
    w0, w1 = max(0, row - 7), min(h, row + 8)
    background, low = float(lrow[w0:w1].max()), float(lrow[row])
    if background - low < 4:
        return None
    half = (background + low) / 2
    a = b = row
    while a > w0 and lrow[a - 1] < half:
        a -= 1
    while b < w1 - 1 and lrow[b + 1] < half:
        b += 1
    return b - a + 1, background - low


def grid_tiles(lum, period):
    """Per tile with a periodic grid: (fwhm 5 mm, contrast 5 mm, fwhm 1 mm, contrast 1 mm)."""
    import warnings
    out = []
    height, width = lum.shape
    for y0 in range(0, height - GRID_TILE_H // 2, GRID_TILE_H):
        for x0 in range(0, width - GRID_TILE_W // 2, GRID_TILE_W):
            tile = lum[y0:y0 + GRID_TILE_H, x0:x0 + GRID_TILE_W]
            if tile.shape[0] < 6 * period or tile.shape[1] < 100:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # rows the stroke mask empties entirely
                lrow = np.nanmedian(tile, axis=1)
            known = np.isfinite(lrow)
            if known.mean() < 0.6:
                continue
            lrow = np.where(known, lrow, 100.0)
            f1, f5, periodicity = grid_phase(lrow, period)
            if periodicity < GRID_MIN_PERIODICITY:
                continue
            major, minor = [], []
            for k in range(-1, int(len(lrow) / period) + 2):
                row = f1 + k * period
                if not 8 <= row <= len(lrow) - 9:
                    continue
                steps = (row - f5) / period
                rule = rule_fwhm(lrow, int(round(row)))
                if rule is not None:
                    is_major = abs(steps - round(steps)) < 0.2 and round(steps) % 5 == 0
                    (major if is_major else minor).append(rule)
            if major:
                out.append((np.median([r[0] for r in major]), np.median([r[1] for r in major]),
                            np.median([r[0] for r in minor]) if minor else np.nan,
                            np.median([r[1] for r in minor]) if minor else np.nan))
    return out


def grid_rules(page400, period, out):
    """grid5_*/grid1_* columns: the horizontal rules on the digitizer's 400 dpi page."""
    out.update({column: "" for column in GRID_COLUMNS})
    lum = cv2.cvtColor(page400, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 2.55
    first = grid_tiles(lum, period)
    if not first:
        out["grid_tiles"] = 0
        return
    # Leave out the strokes darker than any rule of this page - trace, text, handwriting - and
    # 10 px above and below them (the reference drops +-10 px around the decoded trace), then
    # measure again. The threshold follows the page's own rules: a pale grid gets a low one.
    paper = float(np.percentile(lum, 90))
    rule_contrast = float(np.nanpercentile([t[1] for t in first], 90))
    stroke = (paper - lum) > max(30.0, 1.6 * rule_contrast)
    stroke = cv2.dilate(stroke.astype(np.uint8), np.ones((21, 1), np.uint8)) > 0
    tiles = np.array(grid_tiles(np.where(stroke, np.nan, lum), period) or first, float)
    out.update(grid5_fwhm_px400=round(float(np.median(tiles[:, 0])), 2),
               grid5_contrast=round(float(np.median(tiles[:, 1])), 2),
               grid1_fwhm_px400=round(float(np.nanmedian(tiles[:, 2])), 2) if np.isfinite(tiles[:, 2]).any() else "",
               grid1_contrast=round(float(np.nanmedian(tiles[:, 3])), 2) if np.isfinite(tiles[:, 3]).any() else "",
               grid_tiles=len(tiles))


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
    # inc_grid's draw; absent (the upstream 0.5 pt = 0.176 mm on both rules) when the group is off
    widths = data.get("grid_line_width_mm") or {}
    out["truth_grid_major_mm"] = widths.get("major", "")
    out["truth_grid_minor_mm"] = widths.get("minor", "")
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
    if page_rectifier is not None:
        stages = prepare_page(page, page_rectifier)
        out["page400_source"] = stages.source
        if stages.source == "original":  # no scale: the page never reached 400 dpi
            out.update({column: "" for column in GRID_COLUMNS})
        else:
            grid_rules(stages.page, stages.out_px_per_mm, out)
        del stages
    if args.synthetic:
        truth(path, out)
    rows.append(out)
    print("%-28s %s px/mm (%s)  paper %s/%s/%s  cover %s  fwhm %s mm  rules 5/1 mm %s/%s px400" % (
        name[:28], out["px_per_mm"] or "?", out["scale_source"] + ("?" if out["scale_suspect"] == 1 else ""), out["paper_r"], out["paper_g"], out["paper_b"], out["grid_cover"],
        out["trace_fwhm_mm"] or "?", out.get("grid5_fwhm_px400", "-") or "?", out.get("grid1_fwhm_px400", "-") or "?"),
        flush=True)

columns = []
for row in rows:
    columns += [key for key in row if key not in columns]
write_csv(args.out_csv, rows, columns)
usable = sum(bool(r["px_per_mm"]) and r["scale_suspect"] == 0 for r in rows)
print("wrote %s: %d page(s), %d with a usable grid scale, %d suspect (harmonic?), %d without" % (
    args.out_csv, len(rows), usable, sum(r["scale_suspect"] == 1 for r in rows),
    sum(not r["px_per_mm"] for r in rows)))
