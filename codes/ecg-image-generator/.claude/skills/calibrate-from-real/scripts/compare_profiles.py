#!/usr/bin/env python
"""Steps 3 and 5 - set the real profile against the synthetic one, attribute by attribute.

    compare_profiles.py <real.csv> <synth.csv> [--run-dir RUN] [--populations populations.csv
                        --population NAME] [--synth-filter "truth_palette == 7"] [--out table.md]

Both CSVs come from profile_pages.py. populations.csv (slug, population), written from the
audit, restricts the real side to one population: a lot is a mixture, and a pooled median
describes a page that does not exist.

--run-dir leaves out the real pages the manifest drops (exact copies, repeats of an exam),
so an exam counts once. --synth-filter is a pandas query on the synthetic truth_* columns,
to compare a population against the synthetic pages meant to look like it.

Verdict per attribute, from the real p10-p90 against the synthetic p5-p95 (min-max when
fewer than 40 synthetic pages, where p5/p95 would only be the second extremes):
    covered      the real range sits inside what the generator draws
    partial      they overlap, the real range reaches past one end (named)
    NOT COVERED  they do not overlap
A covered attribute is a NECESSARY condition, not a result: the old calibration matched
image statistics and missed the structural causes entirely. The task evaluation decides.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MM = ["dpi", "page_w_mm", "page_h_mm", "trace_fwhm_mm"]
NUMERIC = ["aspect", "dpi", "page_w_mm", "page_h_mm", "abs_skew_deg", "paper_lum", "paper_r", "paper_g", "paper_b",
           "tone_r", "tone_g", "tone_b", "grid_cover", "grid_hue_signed", "colourfulness", "paper_noise",
           "trace_fwhm_mm", "trace_core", "alias_1440", "alias_1024"]

parser = argparse.ArgumentParser()
parser.add_argument("real_csv")
parser.add_argument("synth_csv")
parser.add_argument("--populations")
parser.add_argument("--population")
parser.add_argument("--out")
parser.add_argument("--run-dir")
parser.add_argument("--synth-filter")
args = parser.parse_args()


def load(path):
    frame = pd.read_csv(path)
    frame["abs_skew_deg"] = frame["skew_deg"].abs() if "skew_deg" in frame else np.nan
    # hue is circular: pink sits at 330-350 and red at 0-20, so fold it to (-180, 180]
    frame["grid_hue_signed"] = ((frame["grid_hue_deg"] + 180) % 360) - 180
    suspect = frame["scale_suspect"].fillna(1).astype(int) == 1
    for column in MM:
        if column in frame:
            frame.loc[suspect, column] = np.nan  # a harmonic scale poisons everything in mm
    return frame


real, synth = load(args.real_csv), load(args.synth_csv)
title = "all real pages"
if args.run_dir:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _common import usable_rows
    real = real[real["slug"].isin({r["slug"] for r in usable_rows(args.run_dir)})]
if args.synth_filter:
    synth = synth.query(args.synth_filter)
if args.populations:
    populations = pd.read_csv(args.populations)
    real = real.merge(populations, on="slug", how="left")
    if args.population:
        real = real[real["population"] == args.population]
        title = "population %r" % args.population
if not len(real) or not len(synth):
    raise SystemExit("nothing to compare: %d real page(s), %d synthetic" % (len(real), len(synth)))

KNOWN_B = {"dpi": "check against the recipe's own resolution draw (350-600 dpi in v2), not only this sample",
           "aspect": "the generator draws US letter only (generator-options.md)",
           "page_w_mm": "US letter only; and this is the IMAGE frame, not the sheet - a photo crops it or adds background: take sheet sizes from flat scans",
           "page_h_mm": "US letter only; and this is the IMAGE frame, not the sheet - a photo crops it or adds background: take sheet sizes from flat scans"}
PIXEL_BOUND = {"paper_noise", "grid_cover", "trace_fwhm_mm"}
ra, sb = real["dpi"].dropna(), synth["dpi"].dropna()
dpi_off = len(ra) >= 3 and len(sb) >= 3 and (np.percentile(ra, 10) < sb.min() or np.percentile(ra, 90) > sb.max())
if args.synth_filter:
    title += " vs synthetic where %s" % args.synth_filter
lines = ["# Real vs synthetic profile - %s" % title, "",
         "%d real page(s), %d synthetic. Real p10/p50/p90 against synthetic p5/p50/p95." % (len(real), len(synth)), "",
         "| attribute | real n | real p10 / p50 / p90 | synth n | synth p5 / p50 / p95 | verdict |",
         "|---|---|---|---|---|---|"]
for column in NUMERIC:
    a = real[column].dropna() if column in real else pd.Series(dtype=float)
    b = synth[column].dropna() if column in synth else pd.Series(dtype=float)
    if len(a) < 3 or len(b) < 3:
        lines.append("| %s | %d | - | %d | - | too few pages |" % (column, len(a), len(b)))
        continue
    r10, r50, r90 = np.percentile(a, [10, 50, 90])
    s5, s50, s95 = np.percentile(b, [5, 50, 95]) if len(b) >= 40 else (b.min(), b.median(), b.max())
    if column == "dpi":  # a sample rarely hits the ends of the recipe's randint: allow 2%
        s5, s95 = s5 * 0.98, s95 * 1.02
    if r90 < s5 or r10 > s95:
        verdict = "**NOT COVERED** (real %s the synthetic range)" % ("below" if r90 < s5 else "above")
    elif r10 >= s5 and r90 <= s95:
        verdict = "covered"
    else:
        ends = [end for end, past in (("low", r10 < s5), ("high", r90 > s95)) if past]
        verdict = "partial (real reaches past the %s end)" % " and ".join(ends)
    if column in KNOWN_B and verdict != "covered":
        verdict += " - known (b): %s" % KNOWN_B[column]
    if column in PIXEL_BOUND and dpi_off:
        verdict += " - read off pixels while the dpi is not covered: resolution first"
    lines.append("| %s | %d | %.3g / %.3g / %.3g | %d | %.3g / %.3g / %.3g | %s |" % (
        column, len(a), r10, r50, r90, len(b), s5, s50, s95, verdict))


def shares(series):
    counts = series.dropna().astype(str).value_counts(normalize=True)
    return ", ".join("%s %.0f%%" % (k, 100 * v) for k, v in counts.items()) or "-"


lines += ["", "## Categorical", "", "| attribute | real | synthetic |", "|---|---|---|"]
lines.append("| usable grid scale | %.0f%% | %.0f%% |" % (
    100 * (real["scale_suspect"] == 0).mean(), 100 * (synth["scale_suspect"] == 0).mean()))
if "gate_pass" in real and "gate_pass" in synth:
    lines.append("| rectifier gate passes | %.0f%% | %.0f%% |" % (100 * (real["gate_pass"] == 1).mean(),
                                                               100 * (synth["gate_pass"] == 1).mean()))
for column in [c for c in synth.columns if c.startswith("truth_") and synth[c].nunique() <= 6]:
    lines.append("| synthetic %s | | %s |" % (column[6:], shares(synth[column])))

lines += ["", "paper_noise, grid_cover and trace_fwhm_mm are read off pixels: when the dpi row is not covered, "
          "treat their rows as a consequence of resolution first, and of the paper second.",
          "", "Lead-name position, layout and name vocabulary are not measured here: compare the audit's "
          "control shares (audit_summary.md) with the synthetic truth_* rows above."]
text = "\n".join(lines) + "\n"
print(text)
if args.out:
    open(args.out, "w").write(text)
    print("wrote %s" % args.out)
