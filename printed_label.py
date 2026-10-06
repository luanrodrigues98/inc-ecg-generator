#Lead names printed the way a thermal ECG printer prints them (realism group lead_name_print).
#
#matplotlib text is a perfect vector glyph in one flat colour. What a scan of an INC printout
#shows is a small sans name, 1.8-2.4 mm tall, in the same ink as the trace, made of the print
#head's dots: some dots missing, the ink fading unevenly, the edges soft. The name is therefore
#rasterised here - PIL, at the render resolution - degraded, and placed on the page with
#imshow. matplotlib's agg_filter would have done the degradation in place, but it allocates a
#canvas of the whole page for every filtered artist: ~540 MB per name at a 1200 dpi render.
#
#PrintedLabel exposes the few Text methods ecg_plot uses (get_window_extent, get_position,
#set_y, remove), so the label anticollision and the text_bounding_box code treat both kinds of
#label alike. Its window extent is the tight box of the inked pixels.
import os

import cv2
import numpy as np
from matplotlib.transforms import Bbox
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

FONTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Fonts')

#A thermal head fires dots on a fixed pitch across the paper; 8 dots/mm is the common one.
THERMAL_DOT_MM = 0.125


def _font(font_file, cap_mm, px_per_mm):
    #A PIL font sized so that a capital H is cap_mm tall on the page.
    path = os.path.join(FONTS_DIR, font_file)
    probe = ImageFont.truetype(path, 200)
    top = probe.getbbox('H', anchor='ls')[1]
    size = max(int(round(200 * cap_mm * px_per_mm / -top)), 4)
    return ImageFont.truetype(path, size)


def thermal_print(alpha, strength, px_per_mm, rng):
    #Degrade an ink coverage map [0, 1] like thermal print. strength in [0, 1]:
    #  > 0.1  the glyph is snapped to the head's dot grid, each dot a soft round spot;
    #  dots drop out with probability 0.18*strength and the ink fades by up to 0.6*strength
    #  along a smooth field (~0.25 mm); above 0.6 the strokes also thin by one pixel.
    #A slight blur softens the edges at any strength.
    if strength > 0.1:
        dot = THERMAL_DOT_MM * px_per_mm
        h, w = alpha.shape
        gh, gw = max(int(round(h / dot)), 1), max(int(round(w / dot)), 1)
        dots = cv2.resize(alpha.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA) > 0.45
        dots &= rng.random(dots.shape) >= 0.18 * strength
        alpha = cv2.resize(dots.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        alpha = gaussian_filter(alpha, 0.28 * dot)
        alpha = np.clip((alpha - 0.25) / 0.5, 0.0, 1.0)
        if strength > 0.6:
            alpha = cv2.erode(alpha, np.ones((2, 2), np.uint8), iterations=1)
        field = gaussian_filter(rng.standard_normal(alpha.shape), 0.25 * px_per_mm)
        field /= max(np.abs(field).max(), 1e-9)
        alpha = alpha * (1.0 - 0.6 * strength * (0.5 + 0.5 * field))
    return np.clip(gaussian_filter(alpha, (0.02 + 0.01 * strength) * px_per_mm), 0.0, 1.0)


def render_label(text, font_file, cap_mm, px_per_mm, strength=0.0, rng=None):
    """Ink coverage of `text`, cropped to its inked pixels, and the number of those rows that
    lie below the baseline (0 for a name without descenders)."""
    font = _font(font_file, cap_mm, px_per_mm)
    left, top, right, bottom = font.getbbox(text, anchor='ls')
    pad = int(np.ceil(0.4 * px_per_mm)) + 2
    img = Image.new('L', (right - left + 2 * pad, bottom - top + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad - left, pad - top), text, font=font, fill=255, anchor='ls')
    baseline_row = pad - top
    alpha = np.asarray(img, dtype=np.float64) / 255.0
    if strength > 0:
        alpha = thermal_print(alpha, strength, px_per_mm, rng)
    rows = np.where(alpha.max(axis=1) > 0.02)[0]
    cols = np.where(alpha.max(axis=0) > 0.02)[0]
    if not len(rows):
        #Every dot dropped out: print nothing rather than fail. Keep a 1 px footprint so the
        #annotation still has a box where the name would have been.
        return np.zeros((1, 1)), 0
    alpha = alpha[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return alpha, (rows[-1] + 1) - baseline_row


class PrintedLabel:
    """A lead name drawn from render_label, anchored like ax.text: x is its left edge; y is its
    baseline (va 'baseline') or its centre (va 'center')."""

    def __init__(self, ax, text, x, y, style, px_per_mm, data_per_mm_x, data_per_mm_y, color,
                 rng=None, va='baseline', zorder=3):
        alpha, descent_px = render_label(text, style['font'], style['cap_mm'], px_per_mm,
                                         style.get('thermal', 0.0), rng)
        self.ax = ax
        self.va = va
        self.width = alpha.shape[1] / px_per_mm * data_per_mm_x
        self.height = alpha.shape[0] / px_per_mm * data_per_mm_y
        self.descent = descent_px / px_per_mm * data_per_mm_y
        rgba = np.zeros(alpha.shape + (4,), np.float32)
        rgba[..., :3] = np.asarray(color, dtype=np.float32)[:3]
        rgba[..., 3] = alpha
        self.x, self.y = x, y
        self.image = ax.imshow(rgba, extent=self._extent(), origin='upper', aspect='auto',
                               interpolation='antialiased', zorder=zorder)

    def _extent(self):
        bottom = self.y - self.descent if self.va == 'baseline' else self.y - self.height / 2
        return [self.x, self.x + self.width, bottom, bottom + self.height]

    def get_position(self):
        return self.x, self.y

    def set_y(self, y):
        self.y = y
        self.image.set_extent(self._extent())

    def get_window_extent(self, renderer=None):
        x0, x1, y0, y1 = self._extent()
        return Bbox(self.ax.transData.transform([[x0, y0], [x1, y1]]))

    def remove(self):
        self.image.remove()
