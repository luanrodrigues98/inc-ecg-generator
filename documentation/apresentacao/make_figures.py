"""Figures for the branch presentation in documentation/apresentacao.

Every panel of the deck is a BEFORE/AFTER pair rendered here, on one record and one seed,
so that two images side by side differ only in the parameter the slide is about. The
"before" side is the parameter at its neutral value, which every increment on this branch
verified byte-identical against the commit before it - so it is the pre-change render and
not an approximation of one.

Three steps, each cached on disk so a rerun only does what is missing:

    render  - seeded in-process renders into out/apresentacao_renders/ (gitignored)
    final   - upstream 2ae7d56 (git archive) vs the production YAML, three PTB-XL records
    crop    - identical crops of each pair into documentation/apresentacao/figuras/

    .venv310/bin/python documentation/apresentacao/make_figures.py [render|final|crop|all]

Single-record mode never seeds itself (run_single_file guards seeding with
hasattr(args, 'st') while the argparse dest is start_index), so render() re-seeds all three
RNGs immediately before each call, the way the sweep harnesses under out/*_eval do.
"""

import json
import os
import random
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.dirname(os.path.dirname(HERE))
os.chdir(GEN)
sys.path.insert(0, GEN)

import cv2
import imgaug
import numpy as np
import yaml

from CameraPhotometry.photometry import daylight_gains
from gen_ecg_image_from_data import run_single_file
from gen_ecg_images_from_data_batch import get_parser

RENDERS = os.path.join(GEN, 'out', 'apresentacao_renders')
FIGS = os.path.join(HERE, 'figuras')
RECORD = os.path.join(GEN, 'SampleData', 'PTB_XL_data', '00001_lr')
SEED = 42

UPSTREAM_COMMIT = '2ae7d56'
PTBXL = '/home/luan-rodrigues/faculdade/inc-ecg-generator/sample-data/ptb-xl/1.0.3/records100/00000'
# Records whose production draw (keyed by name) gives the red or pink palette, the 3x4
# layout and no wrinkle texture - found by replaying the runner's randomize block. All six
# are rendered as one batch, because the draws inside run_single_file (crumple amplitude,
# blur) come from a stream that advances record by record; the three shown are picked from
# that batch for the strongest crumpling on a red or pink grid.
FINAL_RECORDS = ('00007_lr', '00144_lr', '00183_lr', '00230_lr', '00293_lr', '00388_lr')
FINAL_PICK = ('00183_lr', '00230_lr', '00293_lr')
PRODUCTION_YAML = os.path.join(GEN, 'batch_ptbxl_3000.calibrated.yaml')

# Every per-image draw off, so a flag is applied at exactly the value written below.
DETERMINISTIC = dict(deterministic_crumple=True, deterministic_blur=True,
                     deterministic_illum=True, deterministic_vignette=True,
                     deterministic_black_point=True, deterministic_white_point=True,
                     illum_azimuth_deg=45.0)

# The augment stage with everything but the step under study switched off: no rotation,
# no crop (lead_bbox forces it to 0), no imgaug noise and no colour temperature.
AUGMENT_QUIET = dict(augment=True, rotate=0, deterministic_noise=True, noise=0,
                     deterministic_temp=True, temperature=0)

WB_MIRED = 40.0
WB_COOL = daylight_gains(-WB_MIRED)
WB_WARM = daylight_gains(+WB_MIRED)
# The exposure the colour and tone panels are drawn at: neutral, not the 0.82 the stage-D
# commits measured at, because the darker page reads as grey on a projector.
EXP_BASE = 1.0
# The cool side of the white balance needs headroom under paper white to show at all.
WB_EXPOSURE = 0.9

# Every panel is drawn on crumpled paper with the red palette (red major rules over pink
# minor ones), so the comparisons look like the photographed sheets the deck is about.
# A spec opts out by overriding a key, as 'flat' does for the crumple slide's ANTES side.
PAPER = dict(standard_grid_color=5, crumple_amplitude=0.8, crumple_scale_cm=8.0)

# tag -> (output dpi, overrides applied on top of the neutral namespace)
SPECS = {
    # Stage A - content
    'neutral':      (200, {}),
    'thick':        (200, dict(trace_thickness_mm=0.45, trace_thickness_jitter=0.6)),
    # The top of the roteiro's ranges: at the production draw (<= 0.8/cm, <= 0.9 mm) the
    # gaps are real but too sparse to read in a slide-sized crop.
    'thick_drop':   (200, dict(trace_thickness_mm=0.45, trace_thickness_jitter=0.6,
                               trace_dropout_rate=1.5, trace_dropout_length_mm=1.5)),
    'layout':       (200, dict(column_gap_mm=2.5, lead_name_gap_mm=10.0)),
    # Stage B - substrate
    'flat':         (200, dict(crumple_amplitude=0.0)),
    # Stage C - optics
    'blur':         (200, dict(blur_sigma=1.5)),
    # Stage D - illumination and photometry
    'illum':        (200, dict(illum_strength=0.8)),
    'vignette':     (200, dict(vignette=0.45)),
    'exp050':       (200, dict(exposure=0.50)),
    'exp115':       (200, dict(exposure=1.15)),
    'imgaug_2778K': (200, dict(AUGMENT_QUIET, temperature=2778)),
    'imgaug_19560K': (200, dict(AUGMENT_QUIET, temperature=19560)),
    'exp082':       (200, dict(exposure=EXP_BASE)),
    'wb_cool':      (200, dict(exposure=WB_EXPOSURE, wb_r=WB_COOL[0], wb_b=WB_COOL[1])),
    'wb_warm':      (200, dict(exposure=WB_EXPOSURE, wb_r=WB_WARM[0], wb_b=WB_WARM[1])),
    'c075':         (200, dict(exposure=EXP_BASE, contrast=0.75)),
    'c135':         (200, dict(exposure=EXP_BASE, contrast=1.35)),
    's050':         (200, dict(exposure=EXP_BASE, saturation=0.5)),
    's150':         (200, dict(exposure=EXP_BASE, saturation=1.5)),
    # Stage E - sensor, at a low delivered dpi where the sampling limit shows
    'ss1_150':      (150, {}),
    'ss2_150':      (150, dict(supersample=2)),
    'imgaug25_150': (150, dict(AUGMENT_QUIET, supersample=2, noise=25)),
    'sn6_150':      (150, dict(supersample=2, sensor_noise=6.0)),
    # Annotations - a deformed, rotated page with the gridpoint lattice stored
    'annot':        (150, dict(supersample=2, crumple_amplitude=1.0, blur_sigma=0.8,
                               exposure=EXP_BASE, vignette=0.3, store_gridpoints=True,
                               **dict(AUGMENT_QUIET, rotate=5))),
}


def render(tag):
    """One seeded render of SPECS[tag]; returns the PNG path, cached by tag."""
    dpi, overrides = SPECS[tag]
    dst = os.path.join(RENDERS, tag + '.png')
    if os.path.exists(dst):
        return dst
    args = get_parser().parse_args(
        ('-i . -o . -se %d -r %d --store_config 2 --lead_bbox --lead_name_bbox'
         % (SEED, dpi)).split())
    for key, value in {**DETERMINISTIC, **PAPER, **overrides}.items():
        if not hasattr(args, key):
            raise SystemExit('%s: unknown flag %s' % (tag, key))
        setattr(args, key, value)
    args.input_file = RECORD + '.dat'
    args.header_file = RECORD + '.hea'
    args.start_index = -1
    args.encoding = os.path.basename(RECORD)
    scratch = os.path.join(RENDERS, '_tmp_' + tag)
    shutil.rmtree(scratch, ignore_errors=True)
    os.makedirs(scratch)
    args.output_directory = scratch
    random.seed(SEED)
    np.random.seed(SEED)
    imgaug.seed(SEED)
    run_single_file(args)
    stem = os.path.join(scratch, os.path.basename(RECORD) + '-0')
    for ext in ('.json', '.gridpoint_xy.npy', '.gridpoint_mask.npy'):
        if os.path.exists(stem + ext):
            shutil.move(stem + ext, os.path.join(RENDERS, tag + ext))
    shutil.move(stem + '.png', dst)
    shutil.rmtree(scratch, ignore_errors=True)
    return dst


def render_all():
    os.makedirs(RENDERS, exist_ok=True)
    for tag in SPECS:
        print('render', tag, flush=True)
        render(tag)


def render_final():
    """Upstream 2ae7d56 against the production chain, on the same three PTB-XL records."""
    inputs = os.path.join(RENDERS, 'final_inputs')
    os.makedirs(inputs, exist_ok=True)
    for name in FINAL_RECORDS:
        for ext in ('.hea', '.dat'):
            link = os.path.join(inputs, name + ext)
            if not os.path.exists(link):
                os.symlink(os.path.join(PTBXL, name + ext), link)

    upstream_root = os.path.join(RENDERS, 'upstream_' + UPSTREAM_COMMIT)
    if not os.path.isdir(upstream_root):
        os.makedirs(upstream_root)
        # Run from the repository root: from a subdirectory, git archive keeps only that
        # subdirectory's prefix inside the tree-ish, which here yields an empty tar.
        repo_root = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=GEN,
                                   check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
        archive = subprocess.run(
            ['git', 'archive', UPSTREAM_COMMIT + ':codes/ecg-image-generator'],
            cwd=repo_root, check=True, stdout=subprocess.PIPE).stdout
        if len(archive) < 1_000_000:
            raise SystemExit('git archive of %s came back empty' % UPSTREAM_COMMIT)
        subprocess.run(['tar', '-x', '-C', upstream_root], input=archive, check=True)

    upstream_out = os.path.join(RENDERS, 'final_upstream')
    if not os.path.isdir(upstream_out):
        # Upstream's own batch driver, which seeds `random` (single-record mode does not),
        # at its defaults. NOT with --wrinkles/--augment: the quilted wrinkle texture and the
        # 50/50 colour temperature draw are the defects the deck shows being replaced, so
        # using them as the baseline would make the old page look more photographic than it
        # was designed to be. The plain render is the same flat page every other ANTES shows.
        subprocess.run([sys.executable, 'gen_ecg_images_from_data_batch.py',
                        '-i', inputs, '-o', upstream_out, '-se', str(SEED),
                        '--max_num_images', str(len(FINAL_RECORDS))],
                       cwd=upstream_root, check=True)

    final_out = os.path.join(RENDERS, 'final_production')
    if not os.path.isdir(final_out):
        with open(PRODUCTION_YAML) as handle:
            config = yaml.safe_load(handle)
        config.update(input_directory=inputs, output_directory=final_out,
                      max_num_images=len(FINAL_RECORDS), skip_existing=False)
        config_path = os.path.join(RENDERS, 'final_production.yaml')
        with open(config_path, 'w') as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        subprocess.run([sys.executable, 'run_batch_from_config.py', config_path],
                       cwd=GEN, check=True)


# Regions of the page as (x0, y0, x1, y1) fractions, so a crop lands on the same paper
# whatever the dpi of the render it is taken from. Placed on the 00001_lr layout.
PAGE = (0.0, 0.0, 1.0, 1.0)
TRACE = (0.55, 0.53, 0.72, 0.66)      # the V2 complex with its deep S wave
LAYOUT = (0.18, 0.35, 0.40, 0.49)     # first column seam of row 1 with its lead names
CRUMPLE = (0.02, 0.15, 0.52, 0.50)    # blank grid over row 1, where the warp reads best
TONE = (0.0, 0.30, 0.45, 0.65)        # rows 1-2 of the first two columns
COLOUR = (0.04, 0.30, 0.26, 0.47)     # lead I over the grid
SENSOR = (0.56, 0.53, 0.70, 0.65)     # V2 at 150 dpi, zoomed until the pixels show
GRAIN = (0.585, 0.545, 0.655, 0.615)  # one V2 complex, closer still: sensor grain is ~3 levels

# output name -> (render tag, region, ('zoom', nearest factor) or ('fit', width px))
CROPS = {
    'a_traco_antes': ('neutral', TRACE, ('zoom', 3)),
    'a_traco_espessura': ('thick', TRACE, ('zoom', 3)),
    'a_traco_falhas': ('thick_drop', TRACE, ('zoom', 3)),
    'a_layout_antes': ('neutral', LAYOUT, ('zoom', 2)),
    'a_layout_depois': ('layout', LAYOUT, ('zoom', 2)),
    'b_amassado_antes': ('flat', CRUMPLE, ('fit', 1100)),
    'b_amassado_depois': ('neutral', CRUMPLE, ('fit', 1100)),
    'c_blur_antes': ('neutral', TRACE, ('zoom', 3)),
    'c_blur_depois': ('blur', TRACE, ('zoom', 3)),
    'd_luz_antes': ('neutral', PAGE, ('fit', 900)),
    'd_luz_iluminacao': ('illum', PAGE, ('fit', 900)),
    'd_luz_vinheta': ('vignette', PAGE, ('fit', 900)),
    'd_exp_050': ('exp050', PAGE, ('fit', 700)),
    'd_exp_100': ('neutral', PAGE, ('fit', 700)),
    'd_exp_115': ('exp115', PAGE, ('fit', 700)),
    'd_temp_2778': ('imgaug_2778K', PAGE, ('fit', 700)),
    'd_temp_19560': ('imgaug_19560K', PAGE, ('fit', 700)),
    'd_wb_frio': ('wb_cool', PAGE, ('fit', 700)),
    'd_wb_quente': ('wb_warm', PAGE, ('fit', 700)),
    'd_tom_antes': ('exp082', TONE, ('fit', 900)),
    'd_tom_c075': ('c075', TONE, ('fit', 900)),
    'd_tom_c135': ('c135', TONE, ('fit', 900)),
    'd_cor_s050': ('s050', COLOUR, ('zoom', 2)),
    'd_cor_antes': ('exp082', COLOUR, ('zoom', 2)),
    'd_cor_s150': ('s150', COLOUR, ('zoom', 2)),
    'e_ss1': ('ss1_150', SENSOR, ('zoom', 5)),
    'e_ss2': ('ss2_150', SENSOR, ('zoom', 5)),
    'e_ruido_imgaug': ('imgaug25_150', GRAIN, ('zoom', 8)),
    'e_ruido_sensor': ('sn6_150', GRAIN, ('zoom', 8)),
}


def cut(image, region, size):
    """Crop a fractional region and size it for the slide."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = region
    sub = image[int(round(y0*h)):int(round(y1*h)), int(round(x0*w)):int(round(x1*w))]
    kind, value = size
    if kind == 'zoom':
        # NEAREST, so the pixels the slide is about stay pixels instead of being smoothed
        # by the resampling a PDF viewer applies anyway.
        return cv2.resize(sub, None, fx=value, fy=value, interpolation=cv2.INTER_NEAREST)
    scale = value/float(sub.shape[1])
    return cv2.resize(sub, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def save(name, image):
    """PNG for pixel zooms, where JPEG ringing would read as noise; JPEG otherwise."""
    if image.shape[1] > 1200 or name.startswith(('a_', 'c_', 'e_', 'd_cor')):
        path = os.path.join(FIGS, name + '.png')
        cv2.imwrite(path, image)
    else:
        path = os.path.join(FIGS, name + '.jpg')
        cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def overlay_annotations(tag, name, width):
    """The stored trace and gridpoint lattice drawn back onto the page they describe."""
    image = cv2.imread(os.path.join(RENDERS, tag + '.png'), cv2.IMREAD_COLOR)
    with open(os.path.join(RENDERS, tag + '.json')) as handle:
        annotation = json.load(handle)
    assert annotation.get('coordinate_order') == 'xy'
    for lead in annotation['leads']:
        points = np.asarray(lead['plotted_pixels'], dtype=np.float64)
        points = points[np.isfinite(points).all(axis=1)]
        cv2.polylines(image, [np.round(points).astype(np.int32).reshape(-1, 1, 2)],
                      False, (255, 200, 0), 2, cv2.LINE_AA)
    lattice = np.load(os.path.join(RENDERS, tag + '.gridpoint_xy.npy'))
    visible = np.load(os.path.join(RENDERS, tag + '.gridpoint_mask.npy'))
    for row in range(lattice.shape[0]):
        # Row 0 in red: it has to sit along the TOP edge of the page.
        colour, radius = ((0, 0, 230), 7) if row == 0 else ((0, 160, 0), 5)
        for col in range(lattice.shape[1]):
            if visible[row, col]:
                x, y = lattice[row, col]
                cv2.circle(image, (int(round(x)), int(round(y))), radius, colour, -1,
                           cv2.LINE_AA)
    return save(name, cut(image, PAGE, ('fit', width)))


def crop_all():
    os.makedirs(FIGS, exist_ok=True)
    for name, (tag, region, size) in CROPS.items():
        image = cv2.imread(os.path.join(RENDERS, tag + '.png'), cv2.IMREAD_COLOR)
        print('crop', save(name, cut(image, region, size)))
    print('crop', overlay_annotations('annot', 'anot_overlay', 1100))

    sources = (('final_upstream', 'final_upstream_%d'),
               ('final_production', 'final_producao_%d'))
    for directory, pattern in sources:
        for index, record in enumerate(FINAL_PICK, start=1):
            path = os.path.join(RENDERS, directory, record + '-0.png')
            if not os.path.exists(path):
                print('crop: missing %s, run the final step first' % path)
                continue
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            print('crop', save(pattern % index, cut(image, PAGE, ('fit', 800))))


if __name__ == '__main__':
    step = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if step in ('render', 'all'):
        render_all()
    if step in ('final', 'all'):
        render_final()
    if step in ('crop', 'all'):
        crop_all()
