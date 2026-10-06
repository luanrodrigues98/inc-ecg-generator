"""Figures for the slides about the INC-like pages (clinical lead order, INC paper, printed
lead names, flatbed scan, pen notes).

Every pair is the same PTB-XL record and seed rendered by run_batch_from_config's
render_record with batch_ptbxl_inc_v2.yaml, all realism groups at p 0 for the ANTES side and
exactly one at p 1 for the DEPOIS side, so the two differ only in that group. The renders
come from ~/.cache/inc-ecg-generator-run/smoke_realism.py, one directory per group:

    z_base z_order z_paper z_print z_scan z_hand   (records 00031_lr 00032_lr 00033_lr)

    .venv310/bin/python documentation/apresentacao/make_figures_inc.py
"""
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'figuras')
RENDERS = os.path.expanduser('~/.cache/inc-ecg-generator-run/smoke_realism')

PAGE = (0.0, 0.0, 1.0, 1.0)
NAMES = (0.21, 0.66, 0.40, 0.80)   # lead name aVF with the traces around it (px 1200-2250, 2850-3450)

# output name -> (render dir, record, region, output width px, nearest zoom factor or None)
CROPS = {
    'f_ordem_antes': ('z_base', '00033_lr', PAGE, 900, None),
    'f_ordem_depois': ('z_order', '00033_lr', PAGE, 900, None),
    'f_papel_antes': ('z_base', '00032_lr', PAGE, 900, None),
    'f_papel_depois': ('z_paper', '00032_lr', PAGE, 900, None),
    'f_nome_antes': ('z_base', '00031_lr', NAMES, 900, None),
    'f_nome_depois': ('z_print', '00031_lr', NAMES, 900, None),
    'f_scan_antes': ('z_base', '00031_lr', PAGE, 900, None),
    'f_scan_depois': ('z_scan', '00031_lr', PAGE, 900, None),
    'f_caneta_antes': ('z_base', '00032_lr', PAGE, 900, None),
    'f_caneta_depois': ('z_hand', '00032_lr', PAGE, 900, None),
}


for name, (directory, record, (x0, y0, x1, y1), width, _) in CROPS.items():
    image = cv2.imread(os.path.join(RENDERS, directory, record + '-0.png'), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    sub = image[int(y0*h):int(y1*h), int(x0*w):int(x1*w)]
    sub = cv2.resize(sub, None, fx=width/sub.shape[1], fy=width/sub.shape[1],
                     interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(FIGS, name + '.jpg'), sub, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(name, sub.shape)
