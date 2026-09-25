#Realism groups: page features that a batch switches on with a probability of their own.
#
#Every feature added to make the pages look like the INC's printouts is a group here, so the
#effect of each can be measured on small dataset variants against a baseline of the same
#size: the baseline sets every p to 0, a variant raises one. The draw is recorded in each
#page's JSON as realism: {group: true/false}, so a mixed batch can also be split by it.
#
#  clinical_lead_order - 3x4 / 6x2 / 12x1 in clinical lead order (config.yaml lead_layouts);
#                        off prints the upstream flat order, as ptbxl_synthetic_4000 did.
#  lead_name_position  - names drawn by --lead_name_position(_single_column) (above, below,
#                        level); off prints every name below its trace, the upstream way.
#  inc_paper           - one of the INC palettes (7 pink strip, 8 CLB orange, 9 CLB white,
#                        with their paper tint) drawn from `palettes`; off keeps whatever
#                        standard_grid_color the batch set.
#  inc_trace           - a thin trace in the INC's colours: colour from `colors`, width
#                        log-uniform over `thickness_mm`; off keeps the batch's trace_color
#                        and trace_thickness_mm.
#  lead_name_print     - names printed like a thermal printer does: a font from `fonts`, a
#                        capital height in `cap_mm` (1.8-2.4 mm measured on the Cardio scans,
#                        against ~2.8 for the upstream 11 pt), the trace's ink, and a thermal
#                        texture of strength drawn in `thermal` (0 clean, 1 heavy: dots on the
#                        head's 0.125 mm pitch, dots dropping out, fading). Off draws the
#                        upstream matplotlib text.
#  scan_look           - the page is a flatbed scan, not a phone photo: flat on the glass
#                        (no crumple, no wrinkles, no side light, no vignette), neutral white
#                        balance, a bright page, a skew of at most `rotate` degrees, little
#                        blur and ~2 levels of noise - all measured on 39 Cardio scans at
#                        600 dpi (paper luminance median 245, p10 236; high-frequency noise
#                        on the paper ~2 levels; skew median 0.2 deg, max 1.6). Off keeps the
#                        photographic chain the rest of the configuration sets.
#  handwriting         - pen notes on the page (handwritten_notes.py): a name, a date, a time,
#                        a signature, a short remark, written with the upstream handwriting
#                        model in one of its eight styles. `n_notes` [min, max] per page, by
#                        `writers` [min, max] hands, each with a pen from `ink` at an `opacity`,
#                        a line of `stroke_mm` and a digit height of `height_mm`, tilted up to
#                        `tilt_deg`, as neat as `bias` (higher is neater); `kinds` weighs what
#                        is written (date, time, name, signature, note) and `regions` where
#                        (header above the leads, margin below or beside them, between_rows,
#                        over_trace - the only one that crosses a trace). `avoid_names` keeps
#                        the ink off the printed lead names. Each note's box goes in the JSON
#                        as handwriting: [...]. Off writes nothing.
#
#A group absent from the block is not drawn and keeps the behaviour the rest of the
#configuration gives it; the JSON then carries no entry for it.
#
#The flags are drawn ONCE PER RECORD, from a stream keyed by (seed, record name) of their
#own, tag 16: the lead order also decides which time window extract_leads cuts for each
#lead, and that cut is shared by all frames of a record. Every group spends one draw,
#configured or not, in the fixed order of GROUPS, so adding or removing a group never
#changes another group's flag.
import copy
import json
import os
import zlib

import numpy as np
from matplotlib.colors import is_color_like

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fonts')

#Order is load bearing: every group spends one draw in this order, so a new group goes at
#the END, or every existing flag of a batch changes.
GROUPS = ('clinical_lead_order', 'lead_name_position', 'inc_paper', 'inc_trace', 'lead_name_print',
          'scan_look', 'handwriting')

#Parameters a group takes besides p, with their defaults - measured on the Cardio scans
#(2026-09-24) where a number is involved.
DEFAULTS = {
    'clinical_lead_order': {},
    'lead_name_position': {},
    'inc_paper': {'palettes': [7, 7, 7, 7, 7, 8, 8, 9, 9]},
    'inc_trace': {'colors': ['#5d4b3c', '#6c574a', '#8c7369', '#b4968c', '#5d4b3c', '#6c574a',
                             '#414141', '#5c5c5c', '#7a7a7a', '#1a3a8f', '#2451b0'],
                  'thickness_mm': [0.05, 0.4]},
    'lead_name_print': {'fonts': ['Verdana.ttf', 'Verdana.ttf', 'Arial.ttf'],
                        'cap_mm': [1.8, 2.4],
                        'thermal': [0.0, 1.0]},
    #exposure: a linear gain; 0.85-0.97 puts white paper at 237-251 (scanned median 245).
    #white_point stays near 1: a scanner does not blow the paper out, and 0.97 was enough to
    #push it to 255 in the first smoke run; contrast likewise stays near 1. rotate: whole degrees, as get_augment draws them. noise: the
    #maximum of get_augment's per-image draw over range(1, noise + 1), in levels.
    #blur_sigma: the maximum, in px of the delivered image, of the per-image draw.
    'scan_look': {'rotate': 1, 'blur_sigma': 0.6, 'exposure': [0.85, 0.97],
                  'contrast': [0.97, 1.05], 'white_point': [0.99, 1.0],
                  'saturation': [0.9, 1.2], 'noise': 2},
    #Not measured yet on the scans: a ballpoint's line (0.25-0.5 mm) and a hand 3-6 mm tall.
    #ink: blue ballpoints twice as often as black ones. A kinds / regions mapping replaces
    #the default whole: a key left out weighs 0.
    'handwriting': {'n_notes': [1, 4], 'writers': [1, 2],
                    'kinds': {'date': 0.25, 'time': 0.15, 'name': 0.25, 'signature': 0.15,
                              'note': 0.2},
                    'regions': {'header': 0.4, 'margin': 0.2, 'between_rows': 0.25,
                                'over_trace': 0.15},
                    'ink': ['#1b2a7c', '#22349a', '#2a3a8c', '#1e2f6e', '#1c1c22', '#2a2a30'],
                    'opacity': [0.75, 0.95], 'height_mm': [3.0, 6.0], 'stroke_mm': [0.25, 0.5],
                    'tilt_deg': 6.0, 'bias': [0.6, 1.5], 'styles': [0, 1, 2, 3, 4, 5, 6, 7],
                    'avoid_names': True},
}


def _range(group, key, value, low_bound, high_bound):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value) or
            value[0] > value[1] or value[0] < low_bound or value[1] > high_bound):
        raise ValueError("realism: %s.%s must be [low, high] with %s <= low <= high <= %s, got %r"
                         % (group, key, low_bound, high_bound, value))
    return [float(value[0]), float(value[1])]


def parse_realism(spec):
    """Validate a realism block and fill in the defaults.

    spec is a mapping {group: p} or {group: {p: ..., <param>: ...}}, or the same as a JSON
    string (the command-line form). Returns {group: {'p': float, <param>: value}} for the
    configured groups, or None when spec is None or empty.
    """
    if spec is None or spec == '' or spec == {}:
        return None
    if isinstance(spec, str):
        spec = json.loads(spec)
    if not isinstance(spec, dict):
        raise ValueError("realism: expected a mapping of group -> p or {p: ...}, got %r" % (spec,))
    parsed = {}
    for group, entry in spec.items():
        if group not in GROUPS:
            raise ValueError("realism: unknown group %r; the groups are %s" % (group, ', '.join(GROUPS)))
        if not isinstance(entry, dict):
            entry = {'p': entry}
        unknown = sorted(set(entry) - {'p'} - set(DEFAULTS[group]))
        if unknown:
            raise ValueError("realism: %s has no parameter(s) %s; it takes p%s"
                             % (group, ', '.join(unknown),
                                ''.join(', ' + k for k in DEFAULTS[group])))
        p = entry.get('p')
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not 0 <= p <= 1:
            raise ValueError("realism: %s.p must be a probability in [0, 1], got %r" % (group, p))
        out = dict(DEFAULTS[group])
        out.update({k: v for k, v in entry.items() if k != 'p'})
        out['p'] = float(p)
        if group == 'inc_paper':
            if not out['palettes'] or not all(isinstance(v, int) and not isinstance(v, bool)
                                              for v in out['palettes']):
                raise ValueError("realism: inc_paper.palettes must be a non-empty list of palette "
                                 "indices, got %r" % (out['palettes'],))
        if group == 'inc_trace':
            if not out['colors'] or not all(is_color_like(c) for c in out['colors']):
                raise ValueError("realism: inc_trace.colors must be a non-empty list of matplotlib "
                                 "colours (quote hex strings in YAML), got %r" % (out['colors'],))
            out['thickness_mm'] = _range(group, 'thickness_mm', out['thickness_mm'], 1e-3, 2.0)
        if group == 'lead_name_print':
            fonts = out['fonts']
            if not fonts or not all(isinstance(f, str) and os.path.exists(os.path.join(FONTS_DIR, f))
                                    for f in fonts):
                raise ValueError("realism: lead_name_print.fonts must name files in %s, got %r"
                                 % (FONTS_DIR, fonts))
            out['cap_mm'] = _range(group, 'cap_mm', out['cap_mm'], 0.5, 10.0)
            out['thermal'] = _range(group, 'thermal', out['thermal'], 0.0, 1.0)
        if group == 'scan_look':
            for key, low, high in (('exposure', 0.05, 4.0), ('contrast', 0.05, 4.0),
                                   ('white_point', 0.5, 1.0), ('saturation', 0.0, 2.0)):
                out[key] = _range(group, key, out[key], low, high)
            for key, low, high in (('rotate', 0, 45), ('noise', 1, 255)):
                if isinstance(out[key], bool) or not isinstance(out[key], int) or not low <= out[key] <= high:
                    raise ValueError("realism: scan_look.%s must be a whole number in [%d, %d], got %r"
                                     % (key, low, high, out[key]))
            if isinstance(out['blur_sigma'], bool) or not isinstance(out['blur_sigma'], (int, float)) \
                    or not 0 <= out['blur_sigma'] <= 10:
                raise ValueError("realism: scan_look.blur_sigma must be in [0, 10] px, got %r"
                                 % (out['blur_sigma'],))
        if group == 'handwriting':
            _parse_handwriting(out)
        parsed[group] = out
    return parsed or None


def _count_range(group, key, value, low_bound, high_bound):
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            not all(isinstance(v, int) and not isinstance(v, bool) for v in value) or
            value[0] > value[1] or value[0] < low_bound or value[1] > high_bound):
        raise ValueError("realism: %s.%s must be [low, high], whole numbers with %d <= low <= high "
                         "<= %d, got %r" % (group, key, low_bound, high_bound, value))
    return [int(value[0]), int(value[1])]


def _weights(group, key, value, names):
    if (not isinstance(value, dict) or not value or set(value) - set(names) or
            not all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0
                    for v in value.values()) or sum(value.values()) <= 0):
        raise ValueError("realism: %s.%s must map some of %s to weights >= 0 that do not all "
                         "vanish, got %r" % (group, key, ', '.join(names), value))
    return {k: float(v) for k, v in value.items()}


def _parse_handwriting(out):
    #Validates in place the parameters of the handwriting group (handwritten_notes.py).
    group = 'handwriting'
    out['n_notes'] = _count_range(group, 'n_notes', out['n_notes'], 1, 20)
    out['writers'] = _count_range(group, 'writers', out['writers'], 1, 5)
    out['kinds'] = _weights(group, 'kinds', out['kinds'], DEFAULTS[group]['kinds'])
    out['regions'] = _weights(group, 'regions', out['regions'], DEFAULTS[group]['regions'])
    if not out['ink'] or not all(is_color_like(c) for c in out['ink']):
        raise ValueError("realism: handwriting.ink must be a non-empty list of matplotlib colours "
                         "(quote hex strings in YAML), got %r" % (out['ink'],))
    out['opacity'] = _range(group, 'opacity', out['opacity'], 0.05, 1.0)
    out['height_mm'] = _range(group, 'height_mm', out['height_mm'], 0.5, 30.0)
    out['stroke_mm'] = _range(group, 'stroke_mm', out['stroke_mm'], 0.05, 2.0)
    out['bias'] = _range(group, 'bias', out['bias'], 0.0, 10.0)
    tilt = out['tilt_deg']
    if isinstance(tilt, bool) or not isinstance(tilt, (int, float)) or not 0 <= tilt <= 45:
        raise ValueError("realism: handwriting.tilt_deg must be in [0, 45] degrees, got %r" % (tilt,))
    out['tilt_deg'] = float(tilt)
    styles = out['styles']
    if not isinstance(styles, (list, tuple)) or not styles or \
            not all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 7 for v in styles):
        raise ValueError("realism: handwriting.styles must be a non-empty list of the model's "
                         "style indices 0-7, got %r" % (styles,))
    if not isinstance(out['avoid_names'], bool):
        raise ValueError("realism: handwriting.avoid_names must be true or false, got %r"
                         % (out['avoid_names'],))


def draw_realism_flags(seed, record_key, realism):
    """{group: bool} for the configured groups of one record; {} when realism is None."""
    if not realism:
        return {}
    rng = np.random.default_rng([abs(int(seed)), int(record_key), 16])
    draws = rng.random(len(GROUPS))
    return {group: bool(draws[k] < realism[group]['p'])
            for k, group in enumerate(GROUPS) if group in realism}


def record_key(header_file):
    """The key the realism draw of a record is made on: its name, without directory or
    extension. extract_leads and run_single_file both derive it, and must agree."""
    return zlib.crc32(os.path.basename(os.path.splitext(header_file)[0]).encode('utf-8'))


def scan_look_args(args, scan, seed, key):
    """A copy of the chain's arguments turned into a flatbed scan's (group scan_look).

    Every photographic stage is switched off or narrowed to what a scanner does; the few
    values drawn per record come from a stream of their own (tag 22), and the per-image
    draws that remain (blur, rotation, noise) keep going through the chain's own code with
    the maxima set here. Values the record drew under randomize: are replaced, not mixed.
    """
    a = copy.copy(args)
    rng = np.random.default_rng([abs(int(seed)), int(key), 22])
    a.crumple_amplitude = 0.0
    a.wrinkles = False
    a.illum_strength = 0.0
    a.vignette = 0.0
    a.blur_sigma = float(scan['blur_sigma'])
    a.deterministic_blur = False
    a.exposure = float(rng.uniform(*scan['exposure']))
    a.exposure_jitter_stops = 0.0
    a.wb_r = a.wb_b = 1.0
    a.wb_mired_jitter = 0.0
    a.contrast = float(rng.uniform(*scan['contrast']))
    a.contrast_jitter_log2 = 0.0
    a.black_point = 0.0
    a.deterministic_black_point = True
    a.white_point = float(rng.uniform(*scan['white_point']))
    a.deterministic_white_point = True
    a.saturation = float(rng.uniform(*scan['saturation']))
    a.saturation_jitter_log2 = 0.0
    a.hue_rotation = 0.0
    a.hue_rotation_jitter_deg = 0.0
    a.rotate = int(scan['rotate'])
    a.noise = int(scan['noise'])
    a.deterministic_noise = False
    a.sensor_noise = 0.0
    a.sensor_noise_jitter_log2 = 0.0
    return a
