"""Figures for the deck's digitizer slides, taken from the inc-ecg-digitizer repository.

Runs in the digitizer's own Python 3.12 environment (ultralytics lives there, not in the
generator's .venv310):

    ~/.cache/pypoetry/virtualenvs/ecg-digitization-tool-*/bin/python \
        documentation/apresentacao/make_figures_digitizer.py

Two sets of panels:

    rectifier - the three panels (original | detected 5 mm crossings | rectified page) of
                one real scan, cut out of the gallery the epoch-10 evaluation wrote
    yolo      - the lead detector trained on the generator's synthetic batch, run on a
                synthetic validation page and on that same real page, boxes only
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'figuras')
DIGITIZER = '/home/luan-rodrigues/faculdade/inc-ecg-digitizer'

# The accepted 300/600 dpi scan with the largest skew (2.85 deg), so the straightening is
# visible, and with no patient identification anywhere on the sheet. Other tilted scans
# carry a handwritten name or a name field and must not go on a slide.
GALLERY_SCAN = 'f71daa98'
# The white strip above the grid carries faint pencil handwriting on this sheet. The top of
# each panel (original, crossings, rectified) is cut away, not blurred, so none of it
# reaches a slide. The rectified panel needs more because the warp moves the strip down.
HANDWRITING_CUT = (0.13, 0.13, 0.21)
YOLO_WEIGHTS = os.path.join(DIGITIZER, 'models', 'leads_obb_synthetic_val.pt')
SYNTHETIC_VAL = os.path.join(DIGITIZER, 'datasets', 'ecg_obb_synthetic_val', 'val.txt')
SYNTHETIC_JSON = os.path.join(DIGITIZER, 'datasets', 'ptbxl_synthetic_4000')
RED_GRID = [255.0, 0.0, 0.0]


def split_gallery(scan):
    """Cut the three panels out of a gallery strip, dropping the burnt-in title line."""
    strip = cv2.imread(os.path.join(DIGITIZER, 'output', 'rectifier_cardio', 'gallery',
                                    scan + '.jpg'))
    grey = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    body = grey[60:]
    # Separator columns are the ones that stay white for the whole height of the body.
    white = (body > 245).mean(axis=0) > 0.97
    panels, start = [], None
    for x, is_white in enumerate(list(white) + [True]):
        if not is_white and start is None:
            start = x
        elif is_white and start is not None:
            if x - start > 200:
                panels.append((start, x))
            start = None
    if len(panels) != 3:
        raise SystemExit('%s: expected 3 panels, found %d' % (scan, len(panels)))
    rows = np.where((grey[:, panels[0][0]:panels[0][1]] < 245).mean(axis=1) > 0.3)[0]
    top = max(int(rows[rows > 40].min()), 40)
    return [strip[top:, x0:x1] for x0, x1 in panels]


def redness(path):
    """Mean of R minus G over the page, in levels: how pink/red the sheet reads."""
    image = cv2.imread(path)
    small = cv2.resize(image, (320, int(320*image.shape[0]/image.shape[1])),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    return float((small[:, :, 2] - small[:, :, 1]).mean())


def pick_synthetic(limit=400):
    """The reddest synthetic validation page with the red palette and the 3x4 layout.

    The palette alone is not enough: exposure, white point and saturation are drawn per
    page, and many red-palette pages come out washed towards white.
    """
    candidates = []
    with open(SYNTHETIC_VAL) as handle:
        for line in list(handle)[:limit]:
            path = line.strip()
            stem = os.path.splitext(os.path.basename(path))[0]
            meta = os.path.join(SYNTHETIC_JSON, stem + '.json')
            if not os.path.exists(meta):
                continue
            with open(meta) as fh:
                info = json.load(fh)
            if info.get('grid_line_color_major') == RED_GRID and \
                    info.get('number_of_columns_in_image') == 4 and not info.get('wrinkles'):
                candidates.append(path)
    if not candidates:
        raise SystemExit('no red 3x4 synthetic page in %s' % SYNTHETIC_VAL)
    return max(candidates, key=redness)


def draw_leads(model, image):
    """Oriented lead boxes drawn as thick outlines, without class names or scores."""
    result = model.predict(image, imgsz=1024, conf=0.25, verbose=False)[0]
    canvas = image.copy()
    thickness = max(2, image.shape[1]//300)
    boxes = result.obb.xyxyxyxy.cpu().numpy() if result.obb is not None else []
    for corners in boxes:
        cv2.polylines(canvas, [corners.astype(np.int32).reshape(-1, 1, 2)], True,
                      (255, 90, 0), thickness, cv2.LINE_AA)
    return canvas, len(boxes)


def main():
    os.makedirs(FIGS, exist_ok=True)
    original, crossings, rectified = [
        panel[int(round(cut*panel.shape[0])):]
        for panel, cut in zip(split_gallery(GALLERY_SCAN), HANDWRITING_CUT)]
    for name, image in (('dig_retif_antes', original), ('dig_retif_pontos', crossings),
                        ('dig_retif_depois', rectified)):
        cv2.imwrite(os.path.join(FIGS, name + '.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(name, image.shape)

    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)
    synthetic_path = pick_synthetic()
    synthetic = cv2.imread(synthetic_path)
    scale = 1200.0/synthetic.shape[1]
    synthetic = cv2.resize(synthetic, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for name, image in (('dig_yolo_sintetica', synthetic), ('dig_yolo_real', original)):
        canvas, count = draw_leads(model, image)
        cv2.imwrite(os.path.join(FIGS, name + '.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(name, canvas.shape, 'boxes', count)
    print('synthetic page', synthetic_path)


if __name__ == '__main__':
    sys.exit(main())
