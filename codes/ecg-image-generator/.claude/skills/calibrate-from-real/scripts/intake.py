#!/usr/bin/env python
"""Step 0 - intake and hygiene of a lot of real ECG images.

    intake.py scan   <lot_dir> <run_dir>
        Decodes every image (reports truncated and unreadable files), hashes the DECODED
        pixels to find exact copies saved under another name or encoding, and writes
            <run_dir>/manifest.csv            one row per file, idx is the number on the sheets
            <run_dir>/thumbs/<slug>.jpg       1000 px thumbnails, as stored (not rotated)
            <run_dir>/orientation_sheet_NN_LOCAL.jpg   numbered thumbnails, 12 per sheet

    intake.py orient <run_dir> "default=270;5=90;17,18=0"
        Writes <run_dir>/orientation.csv (file, rot_cw_deg): how many degrees CLOCKWISE each
        image must turn to be upright. Indices are the idx of the manifest. Use
        "cardio" instead of a spec to take the digitizer's labels/cardio_orientation.csv
        for files that keep their cache names.

    intake.py check  <lot_dir> <run_dir>
        Redraws the sheets with the rotation applied (upright_sheet_NN_LOCAL.jpg), to
        confirm every page reads upright before anything is measured.

    intake.py pairs  <lot_dir> <run_dir>
        Same-sheet CANDIDATES: pages whose dark ink (traces, text, pen) lines up after a
        small shift - the 300 and 600 dpi scans of one sheet, a page saved twice after a
        crop. Writes same_sheet_candidates.csv and one side-by-side image per candidate
        (pair_<a>_<b>_LOCAL.jpg) to confirm BY EYE: same traces beat for beat means same
        exam. Measured: two rescans scored 0.99 and 0.67, unrelated sheets of one printer up
        to 0.61 (they share the printed template), hence the 0.6 cut and the look. It does
        NOT find two photographs of one sheet taken from different angles (0.24-0.55, lost
        among unrelated pages); those are caught in the audit, by the exam's printed or
        handwritten date and time.

    intake.py view <run_dir> 3,17[,22...]
        Puts any pages side by side, upright (view_3_17_LOCAL.jpg) - to compare two pages
        the audit suspects are the same exam, whatever `pairs` scored them.

    intake.py same-exam <run_dir> "7=3;12=9"
        Records the confirmed ones: page #7 is the same exam as #3 (keep the sharper one
        as the right-hand side). Later steps then count the exam once.

The sheets carry patient names: they stay local (hence _LOCAL) and are never published.
"""
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np

from _common import (cardio_orientation, list_images, read_csv, read_image, require_orientation,
                     shrink, slug, upright, write_csv)

COLUMNS = ["idx", "file", "slug", "status", "width", "height", "megapixels", "aspect",
           "bytes", "parent", "pixel_md5", "exact_dup_of", "same_exam_as"]
TILE, PER_ROW, PER_SHEET = 640, 4, 12


def sheet(tiles, out_path):
    cell = TILE + 16
    rows = []
    for start in range(0, len(tiles), PER_ROW):
        row = []
        for idx, img in tiles[start:start + PER_ROW]:
            canvas = np.full((cell + 34, cell, 3), 255, np.uint8)
            small = shrink(img, TILE)
            y0, x0 = 34 + (cell - small.shape[0]) // 2, (cell - small.shape[1]) // 2
            canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
            cv2.putText(canvas, "#%s" % idx, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 200), 2, cv2.LINE_AA)
            row.append(canvas)
        row += [np.full_like(row[0], 255)] * (PER_ROW - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def write_sheets(tiles, run_dir, prefix):
    paths = []
    for n, start in enumerate(range(0, len(tiles), PER_SHEET)):
        path = Path(run_dir) / ("%s_%02d_LOCAL.jpg" % (prefix, n))
        sheet(tiles[start:start + PER_SHEET], path)
        paths.append(path)
    return paths


def scan(lot_dir, run_dir):
    lot_dir, run_dir = Path(lot_dir), Path(run_dir)
    (run_dir / "thumbs").mkdir(parents=True, exist_ok=True)
    rows, tiles, first_by_hash = [], [], {}
    for idx, path in enumerate(list_images(lot_dir)):
        bgr, status = read_image(path)
        rel = str(path.relative_to(lot_dir))
        row = dict(idx=idx, file=rel, slug=slug(path, lot_dir), status=status, bytes=path.stat().st_size,
                   parent=str(Path(rel).parent), width="", height="", megapixels="", aspect="",
                   pixel_md5="", exact_dup_of="", same_exam_as="")
        if bgr is not None:
            h, w = bgr.shape[:2]
            digest = hashlib.md5(np.ascontiguousarray(bgr).tobytes()).hexdigest()
            row.update(width=w, height=h, megapixels=round(w * h / 1e6, 2), aspect=round(w / h, 4),
                       pixel_md5=digest, exact_dup_of=first_by_hash.get(digest, ""))
            first_by_hash.setdefault(digest, rel)
            thumb = shrink(bgr, 1000)
            cv2.imwrite(str(run_dir / "thumbs" / (row["slug"] + ".jpg")), thumb, [cv2.IMWRITE_JPEG_QUALITY, 88])
            tiles.append((idx, thumb))
        rows.append(row)
        print("%4d %-10s %s" % (idx, status, rel), flush=True)
    if not rows:
        raise SystemExit("no image found under %s" % lot_dir)
    write_csv(run_dir / "manifest.csv", rows, COLUMNS)
    sheets = write_sheets(tiles, run_dir, "orientation_sheet")

    bad = [r for r in rows if r["status"] != "ok"]
    dups = [r for r in rows if r["exact_dup_of"]]
    print("\n%d file(s): %d ok, %d truncated (recovered), %d unreadable" % (
        len(rows), sum(r["status"] == "ok" for r in rows), sum(r["status"] == "truncated" for r in rows),
        sum(r["status"] == "unreadable" for r in rows)))
    for r in bad:
        print("  %-10s %s" % (r["status"], r["file"]))
    print("%d exact pixel duplicate(s)" % len(dups))
    for r in dups:
        print("  %s == %s" % (r["file"], r["exact_dup_of"]))
    decoded = [r for r in rows if r["width"]]
    long_side = sorted(max(r["width"], r["height"]) for r in decoded)
    print("canvas: %d landscape, %d portrait; long side %d-%d px (median %d)" % (
        sum(r["width"] >= r["height"] for r in decoded), sum(r["width"] < r["height"] for r in decoded),
        long_side[0], long_side[-1], long_side[len(long_side) // 2]))
    print("NOTE: the canvas shape does not tell the content's orientation - look at the sheets:")
    for path in sheets:
        print("  %s" % path)


def orient(run_dir, spec):
    rows = read_csv(Path(run_dir) / "manifest.csv")
    out = []
    if spec == "cardio":
        known = cardio_orientation()
        missing = [r["file"] for r in rows if Path(r["file"]).name not in known and r["status"] != "unreadable"]
        if missing:
            raise SystemExit("no Cardio orientation label for: %s" % ", ".join(missing[:10]))
        out = [dict(file=r["file"], rot_cw_deg=known.get(Path(r["file"]).name, 0)) for r in rows]
    else:
        default, by_idx = None, {}
        for part in filter(None, (p.strip() for p in spec.split(";"))):
            keys, value = part.split("=")
            value = int(value)
            if value not in (0, 90, 180, 270):
                raise SystemExit("rotation must be 0, 90, 180 or 270, got %s" % value)
            if keys.strip() == "default":
                default = value
            else:
                for key in keys.split(","):
                    by_idx[int(key)] = value
        for r in rows:
            idx = int(r["idx"])
            if idx not in by_idx and default is None:
                raise SystemExit("no rotation for #%d (%s) and no default= in the spec" % (idx, r["file"]))
            out.append(dict(file=r["file"], rot_cw_deg=by_idx.get(idx, default)))
    write_csv(Path(run_dir) / "orientation.csv", out, ["file", "rot_cw_deg"])
    counts = {}
    for r in out:
        counts[r["rot_cw_deg"]] = counts.get(r["rot_cw_deg"], 0) + 1
    print("wrote %s/orientation.csv: %s" % (run_dir, dict(sorted(counts.items()))))


def check(lot_dir, run_dir):
    orientation = require_orientation(run_dir)
    tiles = []
    for r in read_csv(Path(run_dir) / "manifest.csv"):
        if r["status"] == "unreadable":
            continue
        thumb = cv2.imread(str(Path(run_dir) / "thumbs" / (r["slug"] + ".jpg")))
        tiles.append((r["idx"], upright(thumb, orientation[r["file"]])))
    for path in write_sheets(tiles, run_dir, "upright_sheet"):
        print(path)


def ink_map(page):
    """The darkest ~3% of a page - traces, print, pen - as a blurred 512 x 384 map.

    Taken on the max(R,G,B) plane against the local paper level, where a coloured grid rule
    is bright, and cut at a high percentile so that what is left of the grid drops out:
    the grid is the same on every sheet of a printer and would make all of them match.
    """
    page = shrink(page, 1600)
    value = page.max(2)
    closed = cv2.morphologyEx(value, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    darkness = closed.astype(np.int16) - value
    ink = (darkness > max(25, np.percentile(darkness, 97))).astype(np.float32)
    ink = cv2.resize(cv2.dilate(ink, np.ones((3, 3), np.uint8)), (512, 384), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(ink, (0, 0), 2.0)


def pairs(lot_dir, run_dir, threshold=0.6):
    run_dir, threshold = Path(run_dir), float(threshold)
    orientation = require_orientation(run_dir)
    rows = [r for r in read_csv(run_dir / "manifest.csv") if r["status"] != "unreadable" and not r["exact_dup_of"]]
    maps, aspects = {}, {}
    for r in rows:
        thumb = upright(cv2.imread(str(run_dir / "thumbs" / (r["slug"] + ".jpg"))), orientation[r["file"]])
        maps[r["idx"]] = ink_map(thumb)
        aspects[r["idx"]] = thumb.shape[1] / thumb.shape[0]
    found = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if abs(aspects[a["idx"]] / aspects[b["idx"]] - 1) > 0.08:
                continue
            core = maps[b["idx"]][24:-24, 32:-32]  # tolerates a shift of ~6% of the page
            score = float(cv2.matchTemplate(maps[a["idx"]], core, cv2.TM_CCOEFF_NORMED).max())
            if score >= threshold:
                found.append(dict(idx_a=a["idx"], idx_b=b["idx"], file_a=a["file"], file_b=b["file"], score=round(score, 3),
                                  megapixels_a=a["megapixels"], megapixels_b=b["megapixels"]))
    found.sort(key=lambda r: -r["score"])
    write_csv(run_dir / "same_sheet_candidates.csv", found,
              ["idx_a", "idx_b", "score", "file_a", "file_b", "megapixels_a", "megapixels_b"])
    by_idx = {r["idx"]: r for r in rows}
    for r in found[:40]:
        tiles = []
        for idx in (r["idx_a"], r["idx_b"]):
            row = by_idx[idx]
            thumb = upright(cv2.imread(str(run_dir / "thumbs" / (row["slug"] + ".jpg"))), orientation[row["file"]])
            thumb = cv2.resize(thumb, (1000, int(round(thumb.shape[0] * 1000 / thumb.shape[1]))), interpolation=cv2.INTER_AREA)
            cv2.putText(thumb, "#%s  %s MP" % (idx, row["megapixels"]), (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2, cv2.LINE_AA)
            tiles.append(thumb)
        height = max(t.shape[0] for t in tiles)
        tiles = [np.pad(t, ((0, height - t.shape[0]), (0, 0), (0, 0)), constant_values=255) for t in tiles]
        out = run_dir / ("pair_%s_%s_LOCAL.jpg" % (r["idx_a"], r["idx_b"]))
        cv2.imwrite(str(out), np.hstack([tiles[0], np.full((height, 16, 3), 255, np.uint8), tiles[1]]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("#%-4s #%-4s score %.3f  %s" % (r["idx_a"], r["idx_b"], r["score"], out))
    print("%d same-sheet candidate(s) at score >= %.2f among %d page(s) - confirm each by eye, then "
          "intake.py same-exam" % (len(found), threshold, len(rows)))


def view(run_dir, spec):
    run_dir = Path(run_dir)
    orientation = require_orientation(run_dir)
    by_idx = {r["idx"]: r for r in read_csv(run_dir / "manifest.csv")}
    tiles = []
    for idx in (v.strip() for v in spec.split(",") if v.strip()):
        row = by_idx[idx]
        thumb = upright(cv2.imread(str(run_dir / "thumbs" / (row["slug"] + ".jpg"))), orientation[row["file"]])
        thumb = cv2.resize(thumb, (1000, int(round(thumb.shape[0] * 1000 / thumb.shape[1]))), interpolation=cv2.INTER_AREA)
        cv2.putText(thumb, "#%s  %s MP" % (idx, row["megapixels"]), (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 2, cv2.LINE_AA)
        tiles.append(thumb)
    height = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, height - t.shape[0]), (0, 16), (0, 0)), constant_values=255) for t in tiles]
    out = run_dir / ("view_%s_LOCAL.jpg" % "_".join(v.strip() for v in spec.split(",") if v.strip()))
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(out)


def same_exam(run_dir, spec):
    path = Path(run_dir) / "manifest.csv"
    rows = read_csv(path)
    by_idx = {r["idx"]: r for r in rows}
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        dup, keep = (v.strip() for v in part.split("="))
        if dup not in by_idx or keep not in by_idx or dup == keep:
            raise SystemExit("bad pair %r: both sides are idx values of the manifest" % part)
        by_idx[dup]["same_exam_as"] = by_idx[keep]["file"]
        print("  #%s %s  is dropped as a repeat of  #%s %s (kept)" % (dup, by_idx[dup]["file"], keep, by_idx[keep]["file"]))
    write_csv(path, rows, COLUMNS)
    dropped = [r for r in rows if r.get("same_exam_as")]
    print("%d page(s) marked as repeats of an exam; %d page(s) remain for the later steps" % (
        len(dropped), sum(r["status"] != "unreadable" and not r["exact_dup_of"] and not r.get("same_exam_as") for r in rows)))


if __name__ == "__main__":
    commands = {"scan": scan, "orient": orient, "check": check, "pairs": pairs, "view": view, "same-exam": same_exam}
    if len(sys.argv) < 3 or sys.argv[1] not in commands:
        raise SystemExit(__doc__)
    commands[sys.argv[1]](*sys.argv[2:])
