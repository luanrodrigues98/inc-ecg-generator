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
#
#A group absent from the block is not drawn and keeps the behaviour the rest of the
#configuration gives it; the JSON then carries no entry for it.
#
#The flags are drawn ONCE PER RECORD, from a stream keyed by (seed, record name) of their
#own, tag 16: the lead order also decides which time window extract_leads cuts for each
#lead, and that cut is shared by all frames of a record. Every group spends one draw,
#configured or not, in the fixed order of GROUPS, so adding or removing a group never
#changes another group's flag.
import json
import os

import numpy as np
from matplotlib.colors import is_color_like

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fonts')

GROUPS = ('clinical_lead_order', 'lead_name_position', 'inc_paper', 'inc_trace', 'lead_name_print')

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
        parsed[group] = out
    return parsed or None


def draw_realism_flags(seed, record_key, realism):
    """{group: bool} for the configured groups of one record; {} when realism is None."""
    if not realism:
        return {}
    rng = np.random.default_rng([abs(int(seed)), int(record_key), 16])
    draws = rng.random(len(GROUPS))
    return {group: bool(draws[k] < realism[group]['p'])
            for k, group in enumerate(GROUPS) if group in realism}
