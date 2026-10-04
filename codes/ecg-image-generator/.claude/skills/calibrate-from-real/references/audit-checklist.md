# Visual audit checklist

One JSON object per audited page, one per line, in `<run_dir>/audit.jsonl`. The vocabulary
is fixed (`assets/audit_schema.json`) and `audit_summary.py` refuses anything else: counts
are only comparable from one lot to the next if "photo" always means the same thing.

## How to look

For each page of `audit/selection.csv`, read `audit/<slug>/audit.jpg`: one composite. Its
parts, which also exist as single files for a closer look:

1. `detector` - the page AS THE DETECTOR RECEIVES IT (shrunk to its input size with plain
   linear interpolation, no antialiasing) with the leads the pipeline keeps in green
   (name and confidence) and the boxes it throws away in red. Read channel, layout,
   panels, page furniture and page-wide degradations here - and moire, which only exists
   at this scale - and see what the detector made of the page: leads missed, names
   swapped, boxes swallowing a neighbouring row, a calibration pulse taken for a lead.
2. `row ends` - the left 30% and the right 25% of the lead rows, side by side: lead names
   (what they say, where they sit against the trace), the calibration pulse at either end,
   device text printed next to a name.
3. `detail` (native pixels) - grid style, trace colour and weight, JPEG blocks.
4. `header`, `footer` - printer format, speed/gain line, the exam's date and time.

Open `ends.jpg` on its own before deciding `calibration_pulse` and `name_position` when
the composite is not clear: a narrow pulse with faint flanks reads as a dash at composite
scale, and a "CAL" pulse sits at the RIGHT end of the rows on some printers. Open `detail.jpg` on its
own when the grid or the trace looks odd. `page.jpg` is the clean page without boxes.

Write the lines as you go - a few pages at a time, appended to `audit.jsonl` - not all at
the end: a long audit can be interrupted, and `audit_summary.py --partial` summarises what
is there.

Describe what is on the page; what the detector did with it goes in `notes`. Judge
`in_distribution` last, and by this question: **could the detector's job on this page -
finding the 12 leads and naming them - be learnt from pages the generator draws?** Anything
that touches the leads, their names, the layout, the rows or the page geometry counts.
Furniture away from the leads (punch holes in a margin, edge marks, a stamp far from the
traces) does not make a page out of distribution: list it, and answer `yes`. Leave the sheet
size out of this judgement too: the generator only draws US letter, a known gap reported
once for the whole lot. Never judge by how the page scored.

Audit the control pages with the same care as the worst ones. The control sample is where
the frequencies come from; a gap seen only among the worst pages has no frequency yet.

## Fields

Single-valued:

| field | values | what decides |
|---|---|---|
| `channel` | `scan`, `photo`, `scanner_app`, `photocopy`, `screen_photo`, `unsure` | scan: flat, even light, sharp to the edge. photo: perspective, shadow, background, uneven focus. scanner_app: a phone photo flattened and whitened by an app - cropped to the sheet, paper pushed to white, grid thinned. photocopy: black toner, no colour, grid broken into dots or lost. screen_photo: pixel grid, glare, bezel. A photo of a photocopy is `photo`, with `color_mode` and `grid_style` telling the copy |
| `layout` | `3x4_rhythm`, `3x4`, `6x2`, `6x2_rhythm`, `12x1`, `other` | rows x columns of the lead grid; `_rhythm` when a full-width strip runs under it |
| `bordered_panels` | `yes`, `no` | each lead, or each column, sits in a drawn frame |
| `name_vocabulary` | `I_II_III`, `DI_DII_DIII`, `other`, `none_visible` | how the limb leads are printed |
| `name_position` | `above`, `below`, `level`, `cell_corner`, `mixed`, `none_visible` | against the trace's baseline at its start: above it, below it, on it (between pulse and trace), or in the corner of a bordered cell |
| `paper_color` | `white`, `pink`, `orange_salmon`, `blue`, `green`, `grey`, `yellowed`, `other` | the paper BETWEEN the rules as it appears in the image, not the overall tone: the INC pink strip has near-white paper under a dense pink grid, so `white` + `grid_color: red_pink`; mention a strong colour cast in `notes` |
| `grid_color` | `red_pink`, `orange`, `blue`, `green`, `grey_black`, `brown`, `none`, `other` | the rules |
| `grid_style` | `continuous`, `dotted`, `major_only`, `absent` | read it on `detail.jpg` |
| `trace_color` | `black`, `warm_grey`, `blue`, `grey`, `other` | |
| `trace_weight` | `thin`, `medium`, `thick` | against the 1 mm grid: under a fifth of a cell is thin, over a third is thick |
| `calibration_pulse` | `left`, `right`, `both`, `each_segment`, `absent` | where the 1 mV pulse sits on the lead rows; read it on the row ends (`ends.jpg`) |
| `rhythm_lead` | `II`, `I`, `III`, `V1`, `V5`, `other`, `none` | the lead of the full-width strip; `none` when there is no strip |
| `color_mode` | `color`, `black_and_white` | |
| `in_distribution` | `yes`, `partly`, `no` | could the generator have drawn this page with its current options? |

Lists (use `["none"]` when nothing applies):

- `page_furniture`: `printed_header`, `printed_footer`, `device_text` (a model line such as
  "CLB FIA++ N 25" printed among the leads), `field_labels` ("ID:", "Nome:" printed on the
  grid), `handwriting`, `stamp`, `sticker`, `barcode`, `punch_holes`, `edge_marks` (sensor
  or timing marks along an edge), `none`
- `degradations`: `perspective`, `shadow` (soft gradient), `shadow_edged` (a cast shadow
  with an edge: hand, phone), `moire` (beat bands on the detector view or, for the pages
  `ranking.csv` flags, on the extra 1024 px panel - tag it when either shows it; whether
  it is a GAP is decided against the synthetic profile, since generator pages alias too),
  `blur`,
  `low_resolution`, `jpeg_artifacts`, `color_cast`, `folds_creases`, `glare`,
  `cropped_leads`, `background_clutter`, `skew_over_3deg`, `faded_trace`, `rows_overlap`
  (complexes reaching into the neighbouring row), `none`

Optional number:

- `row_pitch_mm` - baseline-to-baseline distance of the lead rows, counted in 5 mm cells
  on the grid (a 3x4 on US letter is ~36 mm; INC strips ~24 mm); counting 1 mm cells on
  `detail.jpg` is fine where the 5 mm rules do not stand out. Only when the cells can be
  counted; `null` otherwise - never a guess.

Free text:

- `slug`, `group` - copied from `selection.csv`.
- `printer_format` - the machine or form, COARSE, so the summary can count it: "CLB FIA++ N
  25", "INC pink strip", "Cardioline ECG200S". Firmware versions and filter settings go in
  `notes`.
- `population` - a short stable name for the kind of page: `pink_strip_scan`,
  `clb_orange_photo`, `bw_photocopy`. Reuse a name already given; a new name is a claim
  that the page belongs to a different population. Channel + paper + printer format is
  usually the right grain.
- `exam_stamp` - the date and time of the exam as read on the page: printed in the footer
  or header on some printers ("12/12/2022 16:02:15"), handwritten on others, `""` when
  none is legible. Two pages with the same stamp are the same exam. Copy the date and
  time only - never the patient's name.
- `notes` - one sentence: what makes the page unlike a synthetic one, or "nothing".

## Example line

```json
{"slug": "a1b2c3", "group": "control", "channel": "scan", "layout": "3x4_rhythm", "bordered_panels": "no", "name_vocabulary": "I_II_III", "name_position": "above", "rhythm_lead": "II", "row_pitch_mm": 25, "paper_color": "white", "grid_color": "red_pink", "grid_style": "continuous", "trace_color": "warm_grey", "trace_weight": "thin", "calibration_pulse": "left", "color_mode": "color", "page_furniture": ["printed_header", "printed_footer", "handwriting"], "degradations": ["none"], "printer_format": "pink strip, header 'ID / Nome', footer '25 mm/s 10 mm/mV'", "population": "pink_strip_scan", "exam_stamp": "01/02/2020 10:20:30", "in_distribution": "yes", "notes": "nothing"}
```
