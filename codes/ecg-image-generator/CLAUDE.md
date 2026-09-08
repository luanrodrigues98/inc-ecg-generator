# CLAUDE.md — ECG image generator (upstream baseline)

Working directory: `codes/ecg-image-generator`.

The tool renders ECG time series from WFDB records onto paper-like backgrounds
and optionally applies distortions (handwritten text, creases, wrinkles,
augmentation) to produce synthetic training data for ECG digitization models.

**Related branches in this repo:**

- `main` — tracks the same upstream commit.
- `feat/distribution-params` — a large fork that adds corpus calibration, a
  config contract, a test suite and a Makefile. **None of that exists here.** If
  you are looking for `calibration/`, `tests/`, `Makefile` or
  `distribution_config.py`, you are on the wrong branch.

## Language policy

**All new code, comments, docstrings and documentation are written in English.**
The upstream codebase is already entirely English, so there is no boundary to
negotiate on this branch — just keep new work in English.

---

## Environment

**Use `./.venv310/bin/python`.** The toolkit requires Python >= 3.9 and < 3.11;
it does not install on 3.12. The venv is gitignored and shared across branches
in this checkout.

```bash
python3.10 -m venv .venv310
.venv310/bin/pip install -r requirements.txt
```

`requirements.txt` is the current dependency set. `environment_droplet.yml` is an
older conda spec pinning different versions (Python 3.9, TensorFlow 2.13,
numpy 1.24) — treat `requirements.txt` as authoritative unless you specifically
need the droplet environment.

`--hw_text` additionally needs the scispaCy model `en_core_sci_sm`, which is
**not** in `requirements.txt` and must be installed separately (see README).
Without it the flag raises `OSError: [E050]`.

## Running

Both entry points `os.chdir` to their own directory at startup, so relative
paths in arguments resolve against the generator root, not your shell's cwd.

```bash
# single record
.venv310/bin/python gen_ecg_image_from_data.py \
    -i SampleData/PTB_XL_data/00001_lr.dat \
    -hea SampleData/PTB_XL_data/00001_lr.hea \
    -o out -se 42 -st 0

# batch over a directory of WFDB pairs
.venv310/bin/python gen_ecg_images_from_data_batch.py \
    -i <dir with .hea/.dat or .hea/.mat> -o out -se 42
```

`SampleData/PTB_XL_data/` holds one versioned record (`00001_lr`) usable for
smoke tests. The README documents every flag; there is no `--help` summary
beyond argparse.

---

## Architecture

Input is a WFDB record pair (`.hea` header + `.dat` or `.mat` signal). Output is
one PNG per 10-second frame, optionally with a sidecar JSON of annotations.

**Call chain:**

`gen_ecg_image_from_data.py` (`run_single_file`) is the core; the batch driver
`gen_ecg_images_from_data_batch.py` just walks a directory and calls it per
record. Inside `run_single_file`:

1. `extract_leads.get_paper_ecg` — loads the record, standardizes lead names,
   slices it into frames, and writes a WFDB copy of what was actually plotted.
2. `ecg_plot.ecg_plot` — renders one frame with matplotlib: grid, traces, lead
   names, calibration pulse, optional printed header. Returns grid sizes and
   fills `json_dict['leads']` with bounding boxes and `plotted_pixels`.
3. `HandwrittenText.generate.get_handwritten` — overlays handwriting (TensorFlow
   + scispaCy; heavy).
4. `CreasesWrinkles.creases.get_creased` — creases as blurred lines, wrinkles as
   a texture multiply.
5. `ImageAugmentation.augment.get_augment` — imgaug pipeline: rotate, Gaussian
   noise, crop, colour temperature. Rotates the annotations to match.
6. Optional QR code stamped into the top-right corner.

**Supporting modules:** `helper_functions.py` (WFDB header parsing, lead
standardization, bounding-box and pixel-coordinate transforms, unit conversion)
and `TemplateFiles/generate_template.py` (builds the printed patient header from
`.hea` comments).

**Configuration:** `config.yaml` holds plotting constants — `paper_len` (10 s),
`abs_lead_step`, the 3×4 column layout `format_4_by_3`, the lead print order
`leadNames_12`, and the lead-separator tick geometry. `template1.json` /
`template2.json` are printed-text layouts.

**Layout note:** in the 3×4 format each column comes from a *different* 2.5 s
window of the signal — column 1 is seconds 0–2.5, column 2 is 2.5–5, and so on.
This is real ECG behaviour, not a bug.

---

## Known traps — verified on this commit

All of the following were reproduced on `27b90f5` with `.venv310` during a
review of this branch. They are genuine upstream defects; the
`feat/distribution-params` branch fixes each of them. Nothing here is fixed on
this branch — work around them or port the fix deliberately.

### Seed reproducibility

`-se/--seed` does not do what it appears to. Measured, same seed twice:

| mode | flags | reproducible? |
|---|---|---|
| batch | plain | **yes** — byte-identical |
| batch | `--augment` | **no** |
| single record | plain | **no** |
| single record | `--augment` | **no** |

Two independent causes:

- **Single-record mode never seeds at all.** `run_single_file` guards seeding
  with `if hasattr(args, 'st')`, but the argparse `dest` for `-st` is
  `start_index`, so `args.st` never exists and the branch is dead. The same dead
  branch is what should set `args.encoding`.
- **`--augment` uses unseeded RNGs.** Only `random` is seeded; `numpy.random`
  and imgaug's global RNG are not, and `iaa.AdditiveGaussianNoise` draws from
  imgaug's.

Font choice also depends on `os.listdir` order (`gen_ecg_image_from_data.py`),
as does wrinkle-file choice (`creases.py`), and `find_records` does not sort
`os.walk`'s directory list — all filesystem-dependent, so results are not
portable across machines even when seeding works.

### `--augment` requires `--store_config`

`--augment` alone crashes with `TypeError: 'NoneType' object is not
subscriptable`. When `store_config` is 0 the code sets `json_dict = None`, but
`get_augment` unconditionally reads `json_dict['leads']`. Either
`--store_config 1` or `--store_config 2` works.

### `--add_qr_code` crashes in single-record mode

`AttributeError: 'Namespace' object has no attribute 'encoding'`. Same dead
`hasattr(args, 'st')` branch as above — the batch driver sets `args.encoding`
per record, so the flag only works there.

### Flags that are silently inert

- `--deterministic_rot` — declared in both parsers, never read. Rotation is
  always `random.randint(-rotate, rotate)` inside `get_augment`.
- `--deterministic_temp` and `-t/--temperature` — declared, never read. Colour
  temperature is always drawn from `range(2000,4000)` or `range(10000,20000)`.

These parse without error and change nothing, which is worse than rejecting them.

### Wrinkle quilting degenerates

`get_creased` calls `quilt(..., block_size=250, num_block=(1,1))`. With a single
block the loop runs once at `y=0, x=0`, where `L2OverlapDiff` returns 0 for every
candidate, so `argmin` always picks index 0. Verified: the result is always the
top-left 250×250 corner of the texture, identical across calls. The minimum-cut
quilting algorithm is entirely bypassed; only the *choice of wrinkle file*
varies.

### Dead code with a latent bug

`creases.randomPatch` calls `random.randint(h - block_size)` with one argument,
which would raise `TypeError` — it is numpy's signature, not the stdlib's.
It never fires because `quilt` calls `randomBestPatch` instead. Do not go
hunting for a crash here; fix it only if you revive the function.

---

## Working notes

- **No test suite on this branch.** There is no `tests/`, no `pytest.ini`, no
  Makefile. Verify changes by generating images and comparing them yourself.
- **`--store_config 2` is the useful annotation level** — it adds the distortion
  parameters actually applied (noise, crop, rotation, temperature, grid colours)
  on top of the geometry. Combine with `--lead_bbox` and `--lead_name_bbox` for
  bounding boxes.
- **Generated JSON is large.** `plotted_pixels` records every rendered sample
  coordinate per lead, which runs to hundreds of KB per image.
- The `out/` directory in this checkout is untracked scratch output, not
  versioned data.
