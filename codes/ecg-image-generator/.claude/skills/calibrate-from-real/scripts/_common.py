"""Shared helpers of the calibrate-from-real scripts.

Every script here runs in the DIGITIZER's Poetry environment (Python 3.12), on CPU:

    cd /home/luan-rodrigues/faculdade/inc-ecg-digitizer
    poetry run python <skill>/scripts/<script>.py ...

The generator's own .venv310 is only for rendering (run_batch_from_config.py).
"""
import csv
import hashlib
import os
import sys
from pathlib import Path

# Before numpy/torch load: with every hardware thread in use these scripts spent more time
# in the kernel than computing, and several of them run side by side during a calibration.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DIGITIZER = Path(os.environ.get("ECG_DIGITIZER_ROOT", "/home/luan-rodrigues/faculdade/inc-ecg-digitizer"))
# scripts -> calibrate-from-real -> skills -> .claude -> generator root (symlinks resolved)
GENERATOR = Path(os.environ.get("ECG_GENERATOR_ROOT", Path(__file__).resolve().parents[4]))
RUN_ROOT = Path(os.path.expanduser("~/.cache/inc-ecg-generator-run/calib"))

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
LEAD_NAMES = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"]
ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def use_digitizer_src():
    """Put the digitizer's src/ on sys.path (its packages have no __init__.py)."""
    src = str(DIGITIZER / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def list_images(lot_dir):
    """Image files under lot_dir, recursive, sorted by relative path."""
    lot_dir = Path(lot_dir)
    return sorted(p for p in lot_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def slug(path, lot_dir):
    """A flat, filesystem-safe id for an image: its relative path without extension."""
    return str(Path(path).relative_to(lot_dir).with_suffix("")).replace(os.sep, "__")


def read_image(path):
    """(bgr or None, status). status: ok | truncated | unreadable.

    A truncated PNG (no IEND chunk) is refused by OpenCV and recovered by PIL; one of the
    Cardio scans is like that at the source. It is still usable, but worth reporting.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is not None:
        return bgr, "ok"
    from PIL import Image, ImageFile
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(str(path)) as handle:
            handle.load()
            rgb = np.asarray(handle.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), "truncated"
    except Exception:  # noqa: BLE001 - anything undecodable is reported, never raised
        return None, "unreadable"
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def upright(bgr, rot_cw_deg):
    """Rotate clockwise by rot_cw_deg (0/90/180/270), the convention of cardio_orientation.csv."""
    rot = int(rot_cw_deg) % 360
    return bgr if rot == 0 else cv2.rotate(bgr, ROTATE[rot])


def shrink(bgr, long_side):
    scale = long_side / max(bgr.shape[:2])
    if scale >= 1:
        return bgr
    return cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, columns=None):
    columns = columns or list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_orientation(run_dir):
    """{file (relative path): rot_cw_deg} from <run_dir>/orientation.csv, or None if absent.

    Structure measured on a sideways page is meaningless, so the callers refuse to run
    without this file rather than assume 0.
    """
    path = Path(run_dir) / "orientation.csv"
    if not path.exists():
        return None
    return {row["file"]: int(row["rot_cw_deg"]) for row in read_csv(path)}


def require_orientation(run_dir):
    orientation = load_orientation(run_dir)
    if orientation is None:
        raise SystemExit("%s/orientation.csv is missing: decide the orientation first "
                         "(intake.py orient), every structural measure depends on it" % run_dir)
    return orientation


def usable_rows(run_dir):
    """Manifest rows that decode and are neither an exact pixel copy of an earlier file nor
    a confirmed repeat of an exam already in the lot: one page per exam."""
    return [row for row in read_csv(Path(run_dir) / "manifest.csv")
            if row["status"] != "unreadable" and not row["exact_dup_of"] and not row.get("same_exam_as")]


def cardio_orientation():
    """{cache file name: rot_cw_deg} for datasets/cardio_s3_cache, from the digitizer's
    hand-made labels/cardio_orientation.csv. The cache prefixes each name with md5(key)[:8]."""
    out = {}
    for row in read_csv(DIGITIZER / "labels" / "cardio_orientation.csv"):
        prefix = hashlib.md5(row["key"].encode()).hexdigest()[:8]
        out["%s__%s" % (prefix, row["filename"])] = int(row["rot_cw_deg"])
    return out


# ---- the lead detector, as the pipeline uses it -------------------------------------------
def load_yolo(model_path):
    from ultralytics import YOLO
    return YOLO(str(model_path))


def detect_leads(model, bgr):
    """Leads the pipeline would keep on this page: {lead name: (quad (4, 2), confidence)}.

    get_image_boxes decides which boxes survive (one per lead, one name per strip), exactly
    as the digitizer does; it returns only the corners, so each confidence is recovered by
    matching the corners back to the raw detections. CPU on purpose: on the ROCm GPU an
    inference next to a training run brings both down.
    """
    use_digitizer_src()
    from digitizer.digitizer import get_image_boxes
    result = model.predict(bgr, device="cpu", verbose=False)[0]
    kept, _ = get_image_boxes(result, model)
    if result.obb is not None:
        corners = result.obb.xyxyxyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
    else:
        xyxy = result.boxes.xyxy.cpu().numpy()
        corners = np.stack([xyxy[:, [0, 1]], xyxy[:, [2, 1]], xyxy[:, [2, 3]], xyxy[:, [0, 3]]], axis=1)
        confs = result.boxes.conf.cpu().numpy()
    out = {}
    for name, quad in kept.items():
        if name == "pulse":
            continue
        quad = np.asarray(quad, float)
        match = np.abs(corners.reshape(len(corners), -1) - quad.reshape(1, -1)).max(axis=1)
        k = int(np.argmin(match)) if len(match) else -1
        # axis-aligned models round the corners to int in get_image_boxes: allow 1 px
        out[name] = (quad, float(confs[k]) if k >= 0 and match[k] <= 1.0 else float("nan"))
    return out, result


def proxy_score(leads):
    """Label-free score of one page: mean confidence of the kept leads x (leads / 12).

    Validated against the true per-image accuracy on dataset_label with best_v2.pt
    (Spearman about +0.82, n=66). It is a ranking signal, not an accuracy estimate, and
    the validation holds for that model only: rerun check_proxy.py when the model changes.
    """
    confs = [conf for _quad, conf in leads.values() if np.isfinite(conf)]
    if not confs:
        return 0.0
    return float(np.mean(confs)) * min(len(leads), 12) / 12.0


def aliasing_score(bgr, size):
    """How much a plain linear shrink of the page to `size` px (what Ultralytics does)
    differs, at low frequency, from a properly averaged one: beat bands - moire - that the
    detector sees and a person looking at a smoothed thumbnail does not. 0 when the page is
    not shrunk. Measured on a 68-page real lot: median 0.4, p90 1.25; the known moire page
    scored 1.9 at 1024 px and 0.75 at 1440 px - moire depends on the input size."""
    h, w = bgr.shape[:2]
    scale = size / max(h, w)
    if scale >= 1:
        return 0.0
    dims = (round(w * scale), round(h * scale))
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    linear = cv2.resize(grey, dims, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    area = cv2.resize(grey, dims, interpolation=cv2.INTER_AREA).astype(np.float32)
    return round(float(cv2.GaussianBlur(linear - area, (0, 0), 3).std()), 3)
