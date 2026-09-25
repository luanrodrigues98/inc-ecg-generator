#Pen notes written on the printed page (realism group handwriting).
#
#The INC's printouts come back from the ward with ink on them: the patient's name, a date
#("27/11/19") and a time ("9:40hs"), a signature, a short remark ("FC 72", "s/ alt"). For a
#lead detector they are distractors - dark strokes next to, or across, the trace and the
#printed lead names. This module writes such notes on the rendered page, before the creases
#and the crumple, so the ink deforms and shades with the paper.
#
#The strokes come from the handwriting model the upstream HandwrittenText stage uses (Graves'
#mixture-density RNN, HandwrittenText/pretrained/model-29, trained on English handwriting),
#driven here without that stage's defects: the model is loaded once per process into a graph
#of its own; its eight writer styles are used; the text is generated here, INC style, not
#picked by scispaCy; the strokes are rasterised in memory at the render resolution, sized in
#millimetres and never stretched; the ink is a coloured, translucent pen, not black pixels;
#and every note leaves a four-corner box in the JSON. The legacy --hw_text stage is left as it
#was.
#
#The model's charset is ASCII without accents (80 symbols), so the Portuguese text is written
#transliterated: "Joao", "Conceicao", "pre-op".
#
#Every draw comes from streams of its own, keyed by (seed, start_index, page): the content
#(tag 23), the pen strokes of each note (tag 24) and the placement (tag 25). The global
#random, numpy and imgaug generators are never touched, so a page with the notes and the same
#page without them differ by the ink alone - and by what the ink does to the page statistics
#that the mean-preserving stages downstream normalise by.
import itertools
import os
import pickle
import zlib

import cv2
import numpy as np
from matplotlib.colors import to_rgb

from ecg_plot import standard_values

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, 'HandwrittenText', 'pretrained', 'model-29')
DATA_DIR = os.path.join(HERE, 'HandwrittenText', 'data')
MODEL_FIELDS = ('coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2', 'rho',
                'finish', 'zero_states')

#The height of a date, a time or a CRM number in each style, in the model's units: the 2-98
#percentile span of the pen's y over ~30 such strings per style (2026-09-25). height_mm is
#this span on the page, so the eight writers write at the size asked for, not at their own.
CAP_UNITS = {0: 3.93, 1: 2.24, 2: 3.72, 3: 3.75, 4: 3.34, 5: 2.58, 6: 4.42, 7: 3.31}

#A sample is kept only if the model said it had finished the text and the pen did what a pen
#does: no pen-down jump longer than MAX_SEGMENT_UNITS (the longest seen in ~280 finished
#samples was 0.78), a height under MAX_HEIGHT_CAPS of the style's, and a width per character
#between WIDTH_PER_CHAR of it (0.43-0.70 measured) - or, under three characters, no wider than
#SHORT_WORD_WIDTH_CAPS and not a dot. The model sometimes stalls after a letter or two, or
#loses the attention and scribbles; a rejected sample is drawn again, up to SAMPLE_ATTEMPTS
#times.
MAX_SEGMENT_UNITS = 1.2
WIDTH_PER_CHAR = (0.25, 1.1)
SHORT_WORD_WIDTH_CAPS = 2.2
MAX_HEIGHT_CAPS = 2.5
SAMPLE_ATTEMPTS = 4
STEPS_PER_CHAR = 60

#Signatures: the same pen, written fast - a low bias and a larger hand.
SIGNATURE_BIAS = (0.05, 0.3)
SIGNATURE_SCALE = (1.2, 1.6)

#Placement, in millimetres of paper. The occupancy grid is coarse on purpose: a note is a few
#millimetres tall, and the grid is checked for every candidate position.
CELL_MM = 0.5
EDGE_MM = 3.0            #keep the ink this far inside the sheet
GAP_MM = 1.0             #between the rows of leads and the header / margin bands
TRACE_CLEAR_MM = 1.0     #notes outside over_trace keep this far from the trace
NAME_CLEAR_MM = 1.0      #and from the printed lead names
NOTE_CLEAR_MM = 2.0      #and from one another
ROW_SPLIT_MM = 4.0       #leads whose baselines differ by more are on different rows
PLACEMENT_TRIES = 60

FIRST_NAMES = ('Maria', 'Jose', 'Ana', 'Joao', 'Antonio', 'Francisco', 'Carlos', 'Paulo', 'Pedro',
               'Lucas', 'Luiz', 'Marcos', 'Gabriel', 'Rafael', 'Francisca', 'Daniel', 'Marcelo',
               'Bruno', 'Eduardo', 'Felipe', 'Raimundo', 'Manoel', 'Andre', 'Fernando', 'Fabio',
               'Gustavo', 'Juliana', 'Adriana', 'Marcia', 'Fernanda', 'Patricia', 'Aline',
               'Sandra', 'Camila', 'Amanda', 'Bruna', 'Leticia', 'Julia', 'Luciana', 'Vanessa',
               'Mariana', 'Claudia', 'Beatriz', 'Rita', 'Sebastiao', 'Conceicao', 'Aparecida',
               'Joana', 'Tereza', 'Helena', 'Severino', 'Geraldo', 'Benedito', 'Lucia')
SURNAMES = ('Silva', 'Santos', 'Oliveira', 'Souza', 'Ferreira', 'Alves', 'Pereira', 'Lima',
            'Gomes', 'Costa', 'Ribeiro', 'Martins', 'Carvalho', 'Almeida', 'Lopes', 'Soares',
            'Fernandes', 'Vieira', 'Barbosa', 'Rocha', 'Dias', 'Nascimento', 'Andrade', 'Moreira',
            'Nunes', 'Marques', 'Machado', 'Mendes', 'Freitas', 'Cardoso', 'Ramos', 'Goncalves',
            'Santana', 'Teixeira', 'Araujo', 'Correia', 'Cavalcanti', 'Monteiro', 'Moura', 'Batista')
REMARKS = ('RS', 'RSR', 's/ alt', 'sem alt', 'ECG pre-op', 'pre-op', 'rev', 'ok', 'OK', 'controle',
           'repetir', 'urgente', 'BRD', 'BRE', 'FA', 'normal', 'amb', 'retorno')

_MODEL = None


def _model():
    """(session, tensors, translation, styles), loaded on the first call of the process.

    The graph is a tf.Graph of its own and the tensors are read from ITS collections: the
    legacy get_handwritten imports the same meta graph into the default graph, so a global
    tf.compat.v1.get_collection would pick up that copy in a process that ran --hw_text.
    One thread, so the sampling is deterministic and does not compete with the render.
    """
    global _MODEL
    if _MODEL is None:
        import tensorflow as tf
        graph = tf.Graph()
        with graph.as_default():
            config = tf.compat.v1.ConfigProto(device_count={'GPU': 0},
                                              intra_op_parallelism_threads=1,
                                              inter_op_parallelism_threads=1)
            session = tf.compat.v1.Session(graph=graph, config=config)
            saver = tf.compat.v1.train.import_meta_graph(MODEL_PATH + '.meta')
            saver.restore(session, MODEL_PATH)
        tensors = {name: graph.get_collection(name)[0] for name in MODEL_FIELDS}
        with open(os.path.join(DATA_DIR, 'translation.pkl'), 'rb') as handle:
            translation = pickle.load(handle)
        with open(os.path.join(DATA_DIR, 'styles.pkl'), 'rb') as handle:
            styles = pickle.load(handle)
        _MODEL = (session, tensors, translation, styles)
    return _MODEL


def _sample_points(text, style, bias, rng):
    """The model's pen for `text` primed with writer `style`: absolute positions (N, 3)
    [x, y, end of stroke] with y pointing down, and whether the model finished the text.
    Upstream's sample_text, drawing from `rng` instead of numpy's global generator."""
    session, t, translation, styles = _model()
    size = len(translation)
    encoded = np.array([translation.get(c, 0) for c in text])
    prime = list(styles[0][style])
    style_text = np.asarray(styles[1][style])
    encoded = np.r_[style_text, encoded]
    sequence_prime = np.eye(size, dtype=np.float32)[style_text]
    sequence_prime = np.concatenate([sequence_prime, np.zeros((1, size))])[None]
    sequence = np.eye(size, dtype=np.float32)[encoded]
    sequence = np.concatenate([sequence, np.zeros((1, size))])[None]
    fetch = [t[k] for k in ('e', 'pi', 'mu1', 'mu2', 'std1', 'std2', 'rho', 'finish')]
    session.run(t['zero_states'])
    coord = prime[0]
    for k in range(1, len(prime)):
        session.run(fetch, feed_dict={t['coordinates']: coord[None, None],
                                      t['sequence']: sequence_prime, t['bias']: bias})
        coord = prime[k]
    coords = [np.array([0.0, 0.0, 1.0])]
    finished = False
    for _ in range(STEPS_PER_CHAR * len(text)):
        e, pi, mu1, mu2, std1, std2, rho, finish = session.run(
            fetch, feed_dict={t['coordinates']: coord[None, None], t['sequence']: sequence,
                              t['bias']: bias})
        weights = pi[0].astype(np.float64)
        g = rng.choice(len(weights), p=weights / weights.sum())
        cross = std1[0, g] * std2[0, g] * rho[0, g]
        x, y = rng.multivariate_normal([mu1[0, g], mu2[0, g]],
                                       [[std1[0, g] ** 2, cross], [cross, std2[0, g] ** 2]])
        coord = np.array([x, y, float(rng.random() < e[0, 0])])
        coords.append(coord)
        if finish[0, 0] > 0.8:
            finished = True
            break
    coords = np.array(coords)
    coords[-1, 2] = 1.0
    points = coords.copy()
    points[:, :2] = np.cumsum(coords[:, :2], axis=0)
    return points, finished


def _split_strokes(points):
    #Pen-down runs. A lone point is no stroke - the model's starting position is one - and
    #upstream's matplotlib plot drew nothing for it either.
    strokes, begin = [], 0
    for end in np.nonzero(points[:, 2] == 1)[0]:
        stroke = points[begin:end + 1, :2]
        if len(stroke) > 1:
            strokes.append(stroke)
        begin = end + 1
    return strokes


def pen_strokes(text, style, bias, rng):
    """The strokes of `text` in model units (y down), or None when every attempt failed.

    The model all but never writes a slash - it is rare in the IAM handwriting it learnt - and
    runs a date together into one scrawl. A text with slashes is therefore written a group at
    a time, the slash between the groups drawn as a stroke of the same pen.
    """
    if '/' in text:
        return _with_slashes(text, style, bias, rng)
    return _sample_word(text, style, bias, rng)


def _word_groups(strokes, lengths, cap):
    #Split a sample of words written with spaces back into its words (`lengths` characters
    #each): cut at clear gaps at least a space wide, choosing the cuts that land nearest to
    #where the words' shares of the characters put them. The widest gaps alone are not
    #enough - a writer often leaves more room inside "2016" than between "02" and "01".
    strokes = sorted(strokes, key=lambda stroke: stroke[:, 0].min())
    clusters, right = [], None
    for stroke in strokes:
        if right is None or stroke[:, 0].min() > right:
            clusters.append([stroke])
        else:
            clusters[-1].append(stroke)
        right = stroke[:, 0].max() if right is None else max(right, stroke[:, 0].max())
    starts = [min(s[:, 0].min() for s in c) for c in clusters]
    ends = [max(s[:, 0].max() for s in c) for c in clusters]
    candidates = [i for i in range(len(clusters) - 1) if starts[i + 1] - ends[i] >= 0.2 * cap]
    if len(candidates) < len(lengths) - 1:
        return None
    total = sum(lengths) + len(lengths) - 1
    left, width = starts[0], ends[-1] - starts[0]
    wanted = [left + width * (sum(lengths[:k + 1]) + k + 0.5) / total for k in range(len(lengths) - 1)]
    best = min(itertools.combinations(candidates, len(lengths) - 1),
               key=lambda cuts: sum(abs((ends[c] + starts[c + 1]) / 2 - w) for c, w in zip(cuts, wanted)))
    groups, begin = [], 0
    for cut in list(best) + [len(clusters) - 1]:
        groups.append([stroke for cluster in clusters[begin:cut + 1] for stroke in cluster])
        begin = cut + 1
    return groups


def _with_slashes(text, style, bias, rng):
    #The groups are sampled together, separated by spaces - the model writes "27 11 19" far
    #better than "27", "11" and "19" apart, one character on its own being where it stalls
    #most - then pulled together around slashes drawn in the gaps.
    cap = CAP_UNITS[style]
    parts = text.split('/')
    words = [part.strip() for part in parts]
    for _ in range(SAMPLE_ATTEMPTS):
        strokes = _sample_word(' '.join(words), style, bias, rng)
        if strokes is None:
            return None
        groups = _word_groups(strokes, [len(word) for word in words], cap)
        if groups is None:
            continue
        middle = float(np.median(np.concatenate(strokes)[:, 1]))
        out, cursor = [], 0.0
        for index, (part, group) in enumerate(zip(parts, groups)):
            if index:
                #A pen line rising to the right, a little taller than the digits either side.
                x0 = cursor + rng.uniform(0.05, 0.2) * cap
                lean, half = rng.uniform(0.3, 0.5) * cap, rng.uniform(0.5, 0.65) * cap
                line = np.c_[np.linspace(x0, x0 + lean, 4),
                             np.linspace(middle + half, middle - half, 4)]
                out.append(line + rng.normal(0, 0.02 * cap, line.shape))
                cursor = x0 + lean + (rng.uniform(0.05, 0.2) + (0.6 if part[:1] == ' ' else 0.0)) * cap
            shift = cursor - min(stroke[:, 0].min() for stroke in group)
            out.extend(stroke + [shift, 0.0] for stroke in group)
            cursor = max(stroke[:, 0].max() for stroke in out)
        return out
    return None


def _sample_word(text, style, bias, rng):
    """pen_strokes for a text the model writes in one go.

    The text is sampled between two spaces. Primed with a writer's style, the model often
    loses the first symbol ("manda" for "Amanda", "uciana" for "Luciana"): a leading space
    gives the priming somewhere to end. And it ends the text when its attention reaches the
    last symbol, which then comes out cut short ("Silv", "1234"): a trailing space puts that
    cut past the text. The pen may leave a dot in either space - a small stroke apart from
    the rest - dropped here. The ends of long texts still get cut now and then.
    """
    cap = CAP_UNITS[style]
    for _ in range(SAMPLE_ATTEMPTS):
        points, finished = _sample_points(' ' + text + ' ', style, bias, rng)
        strokes = _split_strokes(points)
        while len(strokes) > 1:
            last = strokes[-1]
            rest_right = max(s[:, 0].max() for s in strokes[:-1])
            if np.ptp(last, axis=0).max() < 0.35 * cap and last[:, 0].min() > rest_right - 0.1 * cap:
                strokes.pop()
            else:
                break
        while len(strokes) > 1:
            first = strokes[0]
            rest_left = min(s[:, 0].min() for s in strokes[1:])
            if np.ptp(first, axis=0).max() < 0.35 * cap and first[:, 0].max() < rest_left + 0.1 * cap:
                strokes.pop(0)
            else:
                break
        if not finished or not strokes:
            continue
        ink = np.concatenate(strokes)
        jumps = [np.hypot(*np.diff(s, axis=0).T).max() for s in strokes if len(s) > 1]
        width = np.ptp(ink[:, 0])
        if len(text) >= 3:
            wrong_width = not WIDTH_PER_CHAR[0] * cap <= width / len(text) <= WIDTH_PER_CHAR[1] * cap
        else:
            #One or two characters: a '1' is as narrow as a stroke, so the width only catches a
            #stray line; a stall shows as a dot instead - no height, or two digits in one width.
            wrong_width = (width > SHORT_WORD_WIDTH_CAPS * cap or np.ptp(ink[:, 1]) < 0.4 * cap
                           or (len(text) == 2 and width < 0.25 * cap))
        if (jumps and max(jumps) > MAX_SEGMENT_UNITS) or np.ptp(ink[:, 1]) > MAX_HEIGHT_CAPS * cap \
                or wrong_width:
            continue
        return strokes
    return None


def _underline(strokes, cap, rng):
    #The flourish under a signature: a smooth pen line from just left of the ink to past its
    #right end, a little below it, with a wobble.
    ink = np.concatenate(strokes)
    x0, x1 = ink[:, 0].min(), ink[:, 0].max()
    width = x1 - x0
    xs = np.linspace(x0 - 0.1 * width, x1 + rng.uniform(0.05, 0.3) * width, 5)
    ys = ink[:, 1].max() + 0.2 * cap + rng.uniform(-0.15, 0.15, 5) * cap
    ys += np.linspace(0, rng.uniform(-0.3, 0.3) * cap, 5)
    return strokes + [np.c_[xs, ys]]


# ---- text -------------------------------------------------------------------------------------

def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


def _date(rng):
    day, month, year = int(rng.integers(1, 29)), int(rng.integers(1, 13)), int(rng.integers(2012, 2026))
    form = rng.random()
    if form < 0.55:
        return '%02d/%02d/%02d' % (day, month, year % 100)
    if form < 0.8:
        return '%02d/%02d/%04d' % (day, month, year)
    return '%d/%d/%02d' % (day, month, year % 100)


def _time(rng):
    hour = int(rng.choice([7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 0, 3, 5]))
    minute = int(rng.integers(0, 12)) * 5 if rng.random() < 0.7 else int(rng.integers(0, 60))
    form = rng.random()
    if form < 0.35:
        return '%d:%02dhs' % (hour, minute)
    if form < 0.5:
        return '%d:%02dh' % (hour, minute)
    if form < 0.8:
        return '%dh%02d' % (hour, minute)
    return '%d:%02d' % (hour, minute)


def _name(rng):
    first, surname = _pick(rng, FIRST_NAMES), _pick(rng, SURNAMES)
    form = rng.random()
    if form < 0.5:
        return '%s %s' % (first, surname)
    if form < 0.72:
        return '%s %s %s' % (first, _pick(rng, SURNAMES), surname)
    if form < 0.9:
        return '%s. %s' % (first[0], surname)
    return ('%s %s' % (first, surname)).upper()


def _remark(rng):
    form = rng.random()
    if form < 0.2:
        return 'FC %d' % int(rng.integers(48, 121))
    if form < 0.3:
        return '%s %s' % (_pick(rng, ('Dr.', 'Dra.')), _pick(rng, SURNAMES))
    if form < 0.38:
        return 'CRM %05d' % int(rng.integers(10000, 100000))
    if form < 0.43:
        return 'leito %d' % int(rng.integers(1, 40))
    return _pick(rng, REMARKS)


def _signature(rng):
    #What the hand writes when it signs: a few letters of a surname or a pair of initials.
    surname = _pick(rng, SURNAMES)
    if rng.random() < 0.6:
        return surname[:int(rng.integers(3, min(len(surname), 6) + 1))]
    return _pick(rng, FIRST_NAMES)[0] + surname[:int(rng.integers(1, 4))].lower()


TEXT = {'date': _date, 'time': _time, 'name': _name, 'note': _remark, 'signature': _signature}


def _weighted(rng, weights):
    keys = [k for k in weights if weights[k] > 0]
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def plan_notes(cfg, rng):
    """What goes on the page: the writers and, per note, its kind, text, hand and the region
    asked for. Drawn in a fixed order from one stream (tag 23), before any geometry."""
    writers = []
    for _ in range(int(rng.integers(cfg['writers'][0], cfg['writers'][1] + 1))):
        ink = np.array(to_rgb(_pick(rng, cfg['ink']))) * 255 + rng.normal(0, 6, 3)
        writers.append({'style': int(_pick(rng, cfg['styles'])),
                        'bias': float(rng.uniform(*cfg['bias'])),
                        'ink': np.clip(np.rint(ink), 0, 255).astype(int).tolist(),
                        'opacity': float(rng.uniform(*cfg['opacity'])),
                        'stroke_mm': float(rng.uniform(*cfg['stroke_mm'])),
                        'height_mm': float(rng.uniform(*cfg['height_mm']))})
    notes = []
    for _ in range(int(rng.integers(cfg['n_notes'][0], cfg['n_notes'][1] + 1))):
        kind = _weighted(rng, cfg['kinds'])
        writer = int(rng.integers(len(writers)))
        hand = writers[writer]
        note = {'kind': kind, 'text': TEXT[kind](rng), 'writer': writer,
                'height_mm': hand['height_mm'] * float(rng.uniform(0.9, 1.1)),
                'angle_deg': float(rng.uniform(-cfg['tilt_deg'], cfg['tilt_deg'])),
                'region_asked': _weighted(rng, cfg['regions']), 'bias': hand['bias'],
                'underline': False}
        if kind == 'signature':
            note['bias'] = float(rng.uniform(*SIGNATURE_BIAS))
            note['height_mm'] *= float(rng.uniform(*SIGNATURE_SCALE))
            note['underline'] = bool(rng.random() < 0.5)
        notes.append(note)
    return writers, notes


# ---- page geometry ------------------------------------------------------------------------------

def _corners(box):
    if isinstance(box, dict):
        return np.array([box[k] for k in sorted(box, key=int)], dtype=float)
    return np.asarray(box, dtype=float)


def _trace_runs(lead):
    #The plotted samples of a lead as polylines, broken wherever a sample is masked (NaN).
    points = np.asarray(lead.get('plotted_pixels') or [], dtype=float).reshape(-1, 2)
    good = np.isfinite(points).all(axis=1)
    runs, start = [], None
    for i, ok in enumerate(np.r_[good, False]):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append(points[start:i])
            start = None
    return runs


class PageGeometry:
    """Where ink may go, on a grid of CELL_MM cells over the unpadded render.

    trace  - the plotted samples, and the calibration pulse left of each trace;
    names  - the printed lead names (text_bounding_box);
    fixed  - the '25mm/s' and '10mm/mV' ecg_plot prints on every page, and the notes so far.
    """

    def __init__(self, json_dict, px_per_mm):
        self.px_per_mm = px_per_mm
        self.cell = CELL_MM * px_per_mm
        self.width, self.height = float(json_dict['width']), float(json_dict['height'])
        shape = (int(np.ceil(self.height / self.cell)) + 1, int(np.ceil(self.width / self.cell)) + 1)
        self.trace = np.zeros(shape, np.uint8)
        self.names = np.zeros(shape, np.uint8)
        self.fixed = np.zeros(shape, np.uint8)
        self.trace_points = []
        rows, extent = [], []
        for lead in json_dict.get('leads', []):
            runs = _trace_runs(lead)
            for run in runs:
                cv2.polylines(self.trace, [np.round(run / self.cell).astype(np.int32)], False, 1, 1)
            if runs:
                points = np.concatenate(runs)
                self.trace_points.append(points)
                rows.append(float(np.median(points[:, 1])))
                extent.append(points)
            if 'lead_bounding_box' in lead:
                box = _corners(lead['lead_bounding_box'])
                extent.append(box)
                if runs and box[:, 0].min() < points[:, 0].min() - px_per_mm:
                    #The calibration pulse, drawn in the lead's box ahead of its trace.
                    self._fill(self.trace, box[:, 0].min(), box[:, 1].min(), points[:, 0].min(),
                               box[:, 1].max())
            if 'text_bounding_box' in lead:
                box = _corners(lead['text_bounding_box'])
                extent.append(box)
                self._fill(self.names, box[:, 0].min(), box[:, 1].min(), box[:, 0].max(), box[:, 1].max())
        #The scale legend ecg_plot writes at data (2, 0.5) and (4, 0.5): x in seconds at 25 mm/s,
        #y in mV at 10 mm/mV from the bottom, an 11 pt label.
        mm_x = standard_values['x_grid_inch'] * 25.4 / standard_values['x_grid_size']
        mm_y = standard_values['y_grid_inch'] * 25.4 / standard_values['y_grid_size']
        font_mm = standard_values['lead_fontsize'] * 25.4 / 72
        for x_data, chars in ((2, 6), (4, 7)):
            x = x_data * mm_x * px_per_mm
            base = self.height - 0.5 * mm_y * px_per_mm
            self._fill(self.fixed, x - px_per_mm, base - 1.2 * font_mm * px_per_mm,
                       x + (0.7 * chars * font_mm + 1) * px_per_mm, base + 0.4 * font_mm * px_per_mm)
        points = np.concatenate(extent) if extent else np.array([[self.width / 2, self.height / 2]])
        self.left, self.top = points.min(axis=0)
        self.right, self.bottom = points.max(axis=0)
        self.rows = []
        for y in sorted(rows):
            if self.rows and y - self.rows[-1][-1] <= ROW_SPLIT_MM * px_per_mm:
                self.rows[-1].append(y)
            else:
                self.rows.append([y])
        self.rows = [float(np.mean(r)) for r in self.rows]
        self.printed_header = bool(json_dict.get('printed_text'))

    def _fill(self, grid, x0, y0, x1, y1):
        c = self.cell
        grid[max(int(y0 // c), 0):max(int(np.ceil(y1 / c)) + 1, 0),
             max(int(x0 // c), 0):max(int(np.ceil(x1 / c)) + 1, 0)] = 1

    def _dilate(self, grid, mm):
        k = 2 * int(np.ceil(mm / CELL_MM)) + 1
        return cv2.dilate(grid, np.ones((k, k), np.uint8))

    def bands(self, region):
        """Rectangles (x0, y0, x1, y1) in px a note of this region is placed inside."""
        mm = self.px_per_mm
        edge, gap = EDGE_MM * mm, GAP_MM * mm
        w, h = self.width, self.height
        if region == 'header':
            return [] if self.printed_header else [(edge, edge, w - edge, self.top - gap)]
        if region == 'margin':
            return [(edge, self.bottom + gap, w - edge, h - edge),
                    (edge, edge, self.left - gap, h - edge),
                    (self.right + gap, edge, w - edge, h - edge)]
        if region == 'between_rows':
            return [(self.left, a, self.right, b) for a, b in zip(self.rows, self.rows[1:])]
        return []

    def polygon_cells(self, polygon):
        #The grid cells a polygon (4, 2) in px covers, as (row slice, col slice, mask).
        cells = polygon / self.cell
        x0, y0 = np.floor(cells.min(axis=0)).astype(int)
        x1, y1 = np.ceil(cells.max(axis=0)).astype(int) + 1
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(mask, [np.round(cells - [x0, y0]).astype(np.int32)], 1)
        return slice(y0, y1), slice(x0, x1), mask

    def hits(self, grid, polygon):
        rows, cols, mask = self.polygon_cells(polygon)
        patch = grid[max(rows.start, 0):rows.stop, max(cols.start, 0):cols.stop]
        mask = mask[max(-rows.start, 0):, max(-cols.start, 0):][:patch.shape[0], :patch.shape[1]]
        return bool((patch & mask).any())

    def inside(self, polygon):
        edge = EDGE_MM * self.px_per_mm
        return bool((polygon.min(axis=0) >= edge).all() and polygon[:, 0].max() <= self.width - edge
                    and polygon[:, 1].max() <= self.height - edge)

    def place(self, polygon, region, avoid_names, rng):
        """Centre (x, y) in px for a note whose box, centred on the origin, is `polygon`, or
        None when the region has no room for it."""
        forbid_names = self._dilate(self.names, NAME_CLEAR_MM) if avoid_names else 0 * self.names
        forbidden = forbid_names | self.fixed
        half = np.ptp(polygon, axis=0) / 2
        if region == 'over_trace':
            if not self.trace_points:
                return None
            sizes = np.array([len(p) for p in self.trace_points], dtype=float)
            for _ in range(PLACEMENT_TRIES):
                points = self.trace_points[int(rng.choice(len(sizes), p=sizes / sizes.sum()))]
                anchor = points[int(rng.integers(len(points)))]
                centre = anchor + rng.uniform(-0.35, 0.35, 2) * 2 * half * [1, 0.85]
                candidate = polygon + centre
                if self.inside(candidate) and not self.hits(forbidden, candidate) \
                        and self.hits(self.trace, candidate):
                    return centre
            return None
        forbidden = forbidden | self._dilate(self.trace, TRACE_CLEAR_MM)
        bands = [b for b in self.bands(region)
                 if b[2] - b[0] > 2 * half[0] and b[3] - b[1] > 2 * half[1]]
        if not bands:
            return None
        room = np.array([(b[2] - b[0] - 2 * half[0]) * (b[3] - b[1] - 2 * half[1]) for b in bands])
        for _ in range(PLACEMENT_TRIES):
            x0, y0, x1, y1 = bands[int(rng.choice(len(bands), p=room / room.sum()))]
            centre = np.array([rng.uniform(x0 + half[0], x1 - half[0]),
                               rng.uniform(y0 + half[1], y1 - half[1])])
            candidate = polygon + centre
            if self.inside(candidate) and not self.hits(forbidden, candidate):
                return centre
        return None

    def claim(self, polygon):
        rows, cols, mask = self.polygon_cells(polygon)
        k = int(np.ceil(NOTE_CLEAR_MM / CELL_MM))
        mask = np.pad(mask, k)
        mask = cv2.dilate(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        r0, c0 = rows.start - k, cols.start - k
        patch = self.fixed[max(r0, 0):r0 + mask.shape[0], max(c0, 0):c0 + mask.shape[1]]
        patch |= mask[max(-r0, 0):, max(-c0, 0):][:patch.shape[0], :patch.shape[1]]


# ---- ink --------------------------------------------------------------------------------------

def _smooth(stroke, per_segment=4):
    #Catmull-Rom through the pen samples: the model's points are ~0.1 of a letter apart, which
    #at a 1200 dpi render would show as a polygon.
    if len(stroke) < 3:
        return stroke
    p = np.vstack([stroke[:1], stroke, stroke[-1:]])
    t = np.linspace(0, 1, per_segment, endpoint=False)[:, None]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(stroke[-1:])
    return np.vstack(out)


def _layout(strokes, scale, angle_deg, pad):
    """The strokes in px, centred on the ink and turned by angle_deg (positive rises to the
    right), and the box around them: corners TL, TR, BR, BL of the text's own frame."""
    ink = np.concatenate(strokes) * scale
    lo, hi = ink.min(axis=0), ink.max(axis=0)
    centre = (lo + hi) / 2
    a = np.radians(angle_deg)
    turn = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    placed = [(_smooth(s * scale) - centre) @ turn for s in strokes]
    (x0, y0), (x1, y1) = lo - centre - pad, hi - centre + pad
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ turn
    return placed, box


def _ink(image, strokes, box, offset, hand, px_per_mm, rng):
    #Write one note into the page (BGR[A] uint8, modified in place) with the writer's
    #translucent pen: the paper and whatever is printed under it are multiplied by the ink's
    #transmittance.
    thickness = max(int(round(hand['stroke_mm'] * px_per_mm)), 1)
    soften = 0.015 * px_per_mm
    margin = thickness + int(np.ceil(3 * soften)) + 2
    x0, y0 = np.floor(box.min(axis=0) + offset).astype(int) - margin
    x1, y1 = np.ceil(box.max(axis=0) + offset).astype(int) + margin
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, image.shape[1]), min(y1, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    canvas = np.zeros((y1 - y0, x1 - x0), np.uint8)
    for stroke in strokes:
        points = np.round((stroke + offset - [x0, y0]) * 16).astype(np.int32)
        cv2.polylines(canvas, [points], False, 255, thickness, cv2.LINE_AA, shift=4)
    alpha = canvas.astype(np.float32) / 255.0
    if soften > 0.3:
        alpha = cv2.GaussianBlur(alpha, (0, 0), soften)
    #Pressure: a pen lays down more ink here than there, over a millimetre or two.
    coarse = (max(int((y1 - y0) / (0.5 * px_per_mm)), 2), max(int((x1 - x0) / (0.5 * px_per_mm)), 2))
    field = cv2.GaussianBlur(rng.standard_normal(coarse).astype(np.float32), (0, 0), 1.5)
    field /= max(float(np.abs(field).max()), 1e-6)
    field = cv2.resize(field, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    alpha *= hand['opacity'] * (1.0 - 0.25 * (0.5 + 0.5 * field))
    ink = np.array(hand['ink'][::-1], dtype=np.float32) / 255.0
    window = image[y0:y1, x0:x1, :3].astype(np.float32)
    window *= 1.0 - alpha[:, :, None] * (1.0 - ink)
    image[y0:y1, x0:x1, :3] = np.clip(np.rint(window), 0, 255).astype(np.uint8)


def place_notes(json_dict, cfg, resolution, seed=-1, start_index=-1, page_name=''):
    """Everything but the ink: plan the notes of the page `page_name` (the file name its
    streams are keyed by), write their pen strokes and find them room.

    Returns the planned notes (plan_notes' dicts) with 'hand' (the writer) and either
    'region', 'strokes' and 'box' in px of the unpadded render and 'pen' (the stream the ink
    goes on drawing from), or 'region' None and 'drop' - 'pen' when every sample of the
    model was rejected, 'room' when no region had room for the note.
    """
    page_key = zlib.crc32(page_name.encode('utf-8'))
    key = [abs(int(seed)), abs(int(start_index)), page_key]
    writers, notes = plan_notes(cfg, np.random.default_rng(key + [23]))
    px_per_mm = resolution / 25.4
    geometry = PageGeometry(json_dict, px_per_mm)
    placement = np.random.default_rng(key + [25])
    fallback = [r for r in cfg['regions'] if cfg['regions'][r] > 0 and r != 'over_trace']
    for index, note in enumerate(notes):
        hand = note['hand'] = writers[note['writer']]
        style = hand['style']
        strokes = pen_strokes(note['text'], style, note['bias'],
                              np.random.default_rng(key + [24, index]))
        if strokes is None:
            note.update(region=None, drop='pen')
            continue
        pen = np.random.default_rng(key + [24, index, 1])
        if note['underline']:
            strokes = _underline(strokes, CAP_UNITS[style], pen)
        scale = note['height_mm'] * px_per_mm / CAP_UNITS[style]
        pad = hand['stroke_mm'] * px_per_mm / 2
        strokes, box = _layout(strokes, scale, note['angle_deg'], pad)
        #The region asked for, then the others that keep off the trace in a random order
        #weighted like the configuration; over_trace is never a fallback, so the share of
        #notes across the trace stays at most what was asked for.
        regions = [note['region_asked']]
        rest = [r for r in fallback if r != note['region_asked']]
        while rest:
            regions.append(_weighted(placement, {r: cfg['regions'][r] for r in rest}))
            rest.remove(regions[-1])
        for region in regions:
            centre = geometry.place(box, region, cfg['avoid_names'], placement)
            if centre is not None:
                break
        if centre is None:
            note.update(region=None, drop='room')
            continue
        geometry.claim(box + centre)
        note.update(region=region, strokes=[s + centre for s in strokes], box=box + centre, pen=pen)
    return notes


def add_handwriting(input_file, cfg, json_dict, resolution, seed=-1, start_index=-1):
    """Write pen notes on the page `input_file` (in place) and return their annotation.

    cfg is the parsed realism block's handwriting entry; json_dict the page's annotation in
    the unpadded render frame, which this stage reads for where the traces and the printed
    names are; resolution the dpi of the render. Returns the list stored as
    json_dict['handwriting']: per note its text, kind, the region it went to (and the one
    asked for), the box of its ink - four [x, y] corners, TL TR BR BL of the text's own
    frame, in the frame of the other annotations - and how it was written. A note the model
    or the page had no room for is left out.
    """
    notes = [n for n in place_notes(json_dict, cfg, resolution, seed, start_index,
                                    os.path.basename(input_file)) if n['region'] is not None]
    if not notes:
        return []
    image = cv2.imread(input_file, cv2.IMREAD_UNCHANGED)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    #ecg_plot pads the saved page by pad_inches on every side; the annotations are not.
    offset = np.array([(image.shape[1] - int(json_dict['width'])) // 2,
                       (image.shape[0] - int(json_dict['height'])) // 2], dtype=float)
    written = []
    for note in notes:
        hand = note['hand']
        _ink(image, note['strokes'], note['box'], offset, hand, resolution / 25.4, note['pen'])
        written.append({'text': note['text'], 'kind': note['kind'], 'region': note['region'],
                        'region_asked': note['region_asked'],
                        'box': [[round(float(x), 2), round(float(y), 2)] for x, y in note['box']],
                        'ink': hand['ink'], 'opacity': round(hand['opacity'], 3),
                        'height_mm': round(note['height_mm'], 3),
                        'stroke_mm': round(hand['stroke_mm'], 3),
                        'angle_deg': round(note['angle_deg'], 2), 'style': hand['style'],
                        'bias': round(note['bias'], 3), 'writer': note['writer']})
    cv2.imwrite(input_file, image)
    return written
