# What the generator can already draw

Use this to class each gap: **(a)** an option exists - propose a probability or a range
from the measured frequency - or **(b)** nothing draws it - write a feature request.
The source of truth is the code and the current recipe; re-read them when in doubt, they
move faster than this file:

- `batch_ptbxl_inc_v2.yaml` - the current recipe; its comments hold every measurement the
  values came from.
- `realism.py` - `GROUPS`, `DEFAULTS`, and the validation of each group's parameters.
- `config.yaml` - `lead_layouts`, the clinical order per number of columns.
- `run_batch_from_config.py` - `extends:`, `randomize:`, what the runner refuses.

## How a new YAML is written

A calibration never edits a recipe. It writes a NEW file that extends the current one and
lists only what changes (`assets/probe_template.yaml`):

- `extends: <path>` - relative to the new file's own directory, or absolute.
- `randomize:` and `realism:` merge entry by entry with the base; an entry set to `null`
  removes the base's. Every other key replaces the base's value.
- A bare `realism:` (nothing but comments under it) is `null` and replaces the whole block:
  every group off. Always check the `realism (p)` line `preflight.py` prints.
- Unknown keys are a hard error. Keys are argparse destination names.
- `output_directory` is always new.

## Structure

| trait of the real pages | option | where | notes |
|---|---|---|---|
| layout 3x4 + rhythm / 6x2 / 12x1 | `num_columns: {choice: [4, 4, 4, 2, 1]}` | `randomize:` | repeat a value to weight it. 3 columns has no clinical form. A list of another LENGTH re-rolls every `randomize:` key after it (alphabetical) |
| rhythm strip under the grid | `full_mode: II` (fixed key) | top level | one lead for the whole batch, added as an extra row on EVERY layout, 12x1 included. A 12-lead page WITHOUT a rhythm strip cannot be drawn: `extract_leads` replaces any `full_mode` that is not one of the record's leads - `'None'` included - with the first lead, so `'None'` yields a strip in I. (b) |
| leads in clinical order | `clinical_lead_order` | `realism:` | p 1.0 unless the lot shows another order; a different order is (b) |
| lead name above / below / on the baseline | `lead_name_position` group + weights `lead_name_position: 'above:0.6,below:0.4'` and `lead_name_position_single_column: 'level:0.6,above:0.1,below:0.3'` | `realism:` + top level | one convention per page. `level` only fits 12x1. Weights are per page, set them from the control sample's share |
| distance name to baseline | `lead_name_gap_mm`, `lead_name_gap_jitter_mm` | top level | 10 +- 2 mm today |
| name typography (thermal print) | `lead_name_print` (`fonts`, `cap_mm`, `thermal`) | `realism:` | fonts must exist in `Fonts/` |
| name vocabulary DI / DII / DIII | none | - | (b) |
| name in the corner of a bordered cell | none | - | (b) |
| bordered panels around leads or columns | none | - | (b) |
| sheet size and aspect (A4, long strips of about 280 x 110-125 mm) | none from the YAML: the page is US letter, 279 x 216 mm. `ecg_plot` already knows A0-A4 and letter (`papersize_values`), but `run_single_file` always passes an empty `papersize`; strips have no entry at all | - | (b), small for A4, larger for strips. `pad_inches` / `random_padding` only add a margin. The row pitch (letter: ~36 mm on 3x4; real strips: ~24 mm) follows from the height, and with it how often a tall QRS crosses into the next row |
| gap between columns | `column_gap_mm`, `column_gap_jitter_mm` | top level | |
| calibration pulse present | `calibration_pulse` (fraction of pages) | top level | drawn at the START of the first column's rows (left); a pulse at the right end or at both ends is (b). The generator's lead box of a first-column lead takes the pulse in (`pulse_here` in `ecg_plot.py`); nb 4.3 trims it off before training (`_trim_left_to_trace`), so the detector learns boxes WITHOUT the pulse |
| pulse at the right end ("CAL"), after each segment, no pulse on the rhythm strip | none: the pulse is drawn at the start of the first column's rows and of the rhythm strip | - | (b) |
| printed device text (model line next to lead I, "ID:"/"Nome:" labels) | none; the footer "25mm/s 10mm/mV" is printed on every page | - | (b) |
| punch holes, sensor/timing marks on the edge | none | - | (b) |
| printed patient header | `random_print_header` | top level | 0 on purpose: PTB-XL headers carry no demographics, the block renders empty |
| pen notes | `handwriting` (`n_notes`, `kinds`, `regions`, `ink`, `height_mm`, ...) | `realism:` | p 0.7 is the user's estimate, not measured: the audit's `handwriting` share is the measurement |
| stamps, stickers, barcodes | none | - | (b) |

## Paper, grid, trace

| trait | option | where | notes |
|---|---|---|---|
| INC pink strip / CLB orange / CLB white | `inc_paper: {p, palettes: [7, 8, 9]}` | `realism:` | repeat an index to weight it. Measured on 2026-10-02/03 lots: palette 7 matches the pink strips (dE 4-6); the CLB sheets have NEAR-WHITE paper under dense orange rules, closest to palette 9 (dE 3-7) - palette 8's salmon paper (250,206,188) matches no CLB sheet (dE 20-25), its colour is that of the CLB's ruler. Check a CLB population against `--synth-filter "truth_palette == 9"` |
| red, pink, black-grey, blue, green, brown grid on white paper | `standard_grid_color: {choice: [...]}` - 1 brown, 2 pink, 3 blue, 4 green, 5 red, 6 black | `randomize:` | used when `inc_paper` is off for the page. A list of another length re-rolls the keys after it. Blue or green RULES exist; tinted blue or green PAPER does not: (b) |
| dotted grid, major rules only, no minor rules | none | - | (b) |
| thin warm-grey INC trace | `inc_trace: {p, colors, thickness_mm}` | `realism:` | width is log-uniform over `thickness_mm` |
| other trace colours and widths | `trace_color`, `trace_thickness_mm` | `randomize:` | |
| trace dropouts, pressure modulation | `trace_dropout_rate`, `trace_dropout_length_mm`, `trace_thickness_jitter` | `randomize:` | |
| black-and-white page | `random_bw: 1` in a SEPARATE batch, or `saturation` down to 0 | top level / `randomize:` | never mix `random_bw` into a colour batch: any non-zero value bypasses the palettes for every page. Neither reproduces toner (grid broken into dots, traces thickened): a photocopy look is (b) |

## Acquisition channel and degradations

| trait | option | where | notes |
|---|---|---|---|
| flatbed scan vs phone photo | `scan_look: {p, rotate, blur_sigma, exposure, contrast, white_point, saturation, noise}` | `realism:` | p = share of scans in the control sample |
| resolution | `resolution: {randint: [350, 600]}` | `randomize:` | output dpi; RAM and disk grow with dpi^2 |
| skew | `rotate` (max degrees) | top level; `scan_look.rotate` for scans | the exposed corners are filled black, not with a background |
| crumple, creases, wrinkles | `crumple_amplitude`, `crumple_scale_cm`, `wrinkles`, `crease_angle`, `num_creases_*` | | |
| defocus | `blur_sigma` | top level | pixels at render resolution, not mm |
| side light, shadow gradient | `illum_strength`, `illum_azimuth_deg` | top level | a smooth gradient. A CAST shadow with an edge (hand, phone) is (b) |
| exposure, contrast, clipping | `exposure`, `contrast`, `black_point`, `white_point` | `randomize:` | coupled: see lessons |
| colour cast | `wb_r`, `wb_b`, `wb_mired_jitter` (daylight locus only), `saturation`, `hue_rotation` | | a cast off the daylight locus (strong orange or green) is not reachable: (b) |
| vignette | `vignette` | `randomize:` | |
| sensor noise | `noise`, `sensor_noise` | | |
| perspective (keystone), background around the sheet | none | - | (b); `perspective_*` and `background_*` are deferred in the roteiro |
| moire | none | - | (b) |
| JPEG artefacts | none | - | (b); `jpeg_q` is the roteiro's next increment |
| low resolution (under ~150 dpi of paper) | `resolution` | `randomize:` | drawable, but a batch at 100 dpi is another population: its own YAML |
| glare, screen photographs | none | - | (b) |
| page cut by the frame | `crop` | top level | inert while `lead_bbox: true`, which the recipe needs: effectively (b) |
| phone "scanner app" look (whitened paper, boosted contrast) | partly: `exposure`, `white_point`, `contrast` | `randomize:` | a flat white page with crushed grid is (b) |

## Turning a frequency into a value

- A probability comes from the CONTROL sample, with its interval (`audit_summary.md`).
  With 12 control pages a share of 50% is 29-71%: propose the point value, say the
  interval, and do not move a p that already sits inside it.
- Do not set a p to 0 or 1 because the control sample showed none or all: 0 of 12 is
  still compatible with 20%. Keep some mass unless the trait is structurally impossible.
- A range (trace width, paper luminance, noise) comes from the profile: cover the real
  p10-p90 without going far past it.
- When the lot holds several populations, the lot-level shares become the SIZES of the
  per-population YAMLs (`max_num_images`), and inside each YAML the population's traits
  are pinned.
- The new lot is not the only real data. Unless the user says the generator should now
  target only this lot, the proposal ADDS coverage; it does not narrow the recipe to it.
