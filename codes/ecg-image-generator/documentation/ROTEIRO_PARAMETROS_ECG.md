# Roteiro incremental de parâmetros — gerador de imagens de ECG

Cada seção é um incremento fechado: um parâmetro, sua posição na cadeia, os
atributos do notebook que ele move e um prompt pronto para o plan mode.

A ordem segue a cadeia de formação de imagem definida em `PARAMETROS_GERADOR.md`,
com quatro estágios novos encaixados nas posições fisicamente corretas: espessura
e falhas do traço são **conteúdo** (antes de tudo), amassado é **substrato**, e
gradiente de iluminação entra **junto com a exposição, em luz linear**.

```
content (trace + dropouts) → paper crumple → optics → illumination → exposure
  → white balance → vignette → tone curve → clipping → color → resolution
  → sensor noise → JPEG
```

> **Convenção de idioma.** Todo o código é escrito em inglês: nomes de parâmetros,
> variáveis, funções, classes, comentários, docstrings, mensagens de log e de erro,
> nomes de arquivo e de teste. A documentação e as conversas ficam em português.
> Cada prompt abaixo já abre com essa instrução — se você adaptar algum deles,
> mantenha essa linha.
>
> O módulo `image_pipeline.py` (antes `gerador_calibrado.py`) já está inteiramente
> em inglês e é a referência de nomenclatura. Os nomes de parâmetro deste roteiro
> são os nomes que devem aparecer no código, literalmente.

---

## Regras que valem para todos os incrementos

**Todo parâmetro novo nasce com um valor neutro que é no-op.** `trace_dropout_rate=0`,
`crumple_amplitude=0`, `illum_strength=0` têm que reproduzir a saída atual bit a bit.
Isso dá um teste de regressão de graça a cada passo, e sem ele um bug introduzido no
incremento 3 só aparece no 11.

**Nunca calibre dois parâmetros contra a mesma métrica.** Amassado, gradiente de
iluminação e vinheta os três mexem em `lum_spatial_std`. Se os três forem resolvidos
contra ela, o solver oscila — foi exatamente assim que a calibração divergiu com
85 % de erro no teste anterior. A separação por escala espacial já está implementada
em `measure()`: `lum_spatial_std_low` (grade 3×3) e `lum_spatial_std_mid` (12×12).

**O conteúdo tem que nascer mais limpo e mais nítido que o P1 da distribuição alvo.**
A cadeia só degrada: adiciona ruído, desfoca, comprime. Ela não remove nada. Renderize
o traço nítido e sem ruído; toda a degradação vem depois.

**Valide cada incremento isoladamente:** varra o parâmetro novo em 5 valores com
todo o resto fixo, rode o extrator e confirme que o atributo alvo se move de forma
monotônica e que os demais ficam aproximadamente parados. Se um parâmetro mexe em
tudo, ele está na posição errada da cadeia.

---

# Estágio A — Conteúdo

## 1. `trace_thickness_mm`

**Faixa sugerida:** 0,20 – 0,60 mm (em mm de papel, não em pixels)
**Atributos que move:** `sharp_laplacian_var`, `sharp_tenengrad`, `gradient_mean`, `edge_density`, `glcm_contrast`, `lbp_flat_share`
**Por que primeiro:** é a única propriedade geométrica intrínseca do traço. Tudo
depois dela só a degrada, então ela define o teto de nitidez que a cadeia pode entregar.

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: generator of synthetic images of printed ECG paper, photographed. The
trace is rasterised from a 1D signal onto millimetre grid paper.

Task: add a `trace_thickness_mm` parameter, the trace stroke width in MILLIMETRES
of paper, converted to pixels by the render scale (px/mm). Useful range
0.20-0.60 mm; the current behaviour becomes the default so output does not change.

Requirements:
- Thickness in mm, never in pixels: it must be invariant to render resolution.
- Modulate thickness along the trace with smooth 1D noise (about +/-15%),
  imitating variable stylus pressure. A perfectly constant stroke width is the
  single strongest giveaway of a synthetic ECG.
- Render the trace with anti-aliasing. Without it, `sharp_laplacian_var` is pinned
  to the aliasing level and will not respond to the parameter.
- Apply ONLY to the ECG trace. Grid lines and text have their own stroke width and
  must not be affected.

Validation: render at 0.2/0.3/0.4/0.5/0.6 mm, extract the features, and confirm
that `gradient_mean` and `edge_density` increase monotonically while `lum_mean`
varies by less than 3% across the range.

Produce a plan before writing any code.
```

## 2. `trace_dropout_rate` / `trace_dropout_length_mm`

**Faixa sugerida:** taxa 0 – 1,5 falhas/cm; comprimento 0,1 – 1,5 mm
**Atributos que move:** `edge_density`, `lbp_entropy`, `glcm_contrast`, `glcm_homogeneity`
**Por que aqui:** a falha é ausência de tinta no papel. Precisa existir antes da
óptica para ser desfocada junto com o resto — uma falha aplicada depois do blur
tem borda dura e não parece falha de impressão.

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: same ECG generator. `trace_thickness_mm` already exists.

Task: add intermittent trace dropouts, imitating a stylus losing contact or ink
failing. Two parameters: `trace_dropout_rate` (dropouts per cm of trace, 0-1.5)
and `trace_dropout_length_mm` (mean length, 0.1-1.5 mm). Both default to 0,
reproducing current output exactly.

Requirements:
- Define dropouts along the ARC LENGTH of the trace, not along the x axis. On a
  near-vertical R wave the trace covers a lot of arc in very little x, and a
  dropout must be able to land mid-upstroke.
- Not every dropout is total: sample between a full gap (trace disappears) and
  partial fading (opacity drops to 0.3-0.7). Partial fading is the more common
  case in real thermal printing.
- Dropout lengths from an exponential distribution, not uniform: short gaps are
  far more common than long ones.
- Apply before rasterisation/anti-aliasing so gap edges are smoothed like the rest
  of the trace.
- Never place dropouts over the calibration pulse (the rectangular step at the
  start of each row), which is printed differently.

Validation: sweep `trace_dropout_rate` over 0/0.5/1.0/1.5 and confirm
`edge_density` rises (more discontinuous edges) while `glcm_homogeneity` falls.
Inspect visually at high zoom: dropout edges must not be aliased.

Produce a plan before writing any code.
```

---

# Estágio B — Substrato

## 3. `crumple_amplitude` / `crumple_scale_cm`

**Faixa sugerida:** amplitude 0 – 1,0; escala espacial 3 – 15 cm
**Atributos que move:** `lum_spatial_std_mid`, `glcm_contrast`, `glcm_correlation`, `lum_entropy`
**Por que aqui:** o amassado deforma o papel *e* cria sombreamento. Precisa vir antes
da óptica e da iluminação, porque é a geometria da superfície que a luz vai encontrar.

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator. Trace and dropouts are implemented. The sheet is currently
perfectly flat.

Task: add paper crumpling and waviness, with two parameters: `crumple_amplitude`
(0-1.0, default 0) and `crumple_scale_cm` (3-15).

Core requirement: the geometric deformation and the shading must come from the
SAME height field. If generated independently, creases will not line up with
shadows and the result reads as an overlaid texture rather than crumpled paper.

Suggested implementation:
1. Generate a smooth height field H(x,y) — sum of 2-4 octaves of filtered Gaussian
   noise, with the dominant octave at `crumple_scale_cm`.
2. Deformation: displace pixels along the gradient of H (cv2.remap with a
   displacement field proportional to grad H). Amplitude of a few pixels.
3. Shading: compute surface normals from grad H and apply a Lambertian N.L term
   with a light direction L, multiplying the image IN LINEAR LIGHT (undo the sRGB
   gamma first, redo it after). Shading applied in display space has the wrong
   contrast in dark areas.
4. Reuse the same light direction L in increment 5 (illumination gradient) by
   storing it on the parameter object — two different light directions in one
   image is physically impossible and visually noticeable.

The real reference photo has a soft crease across the upper third and slight
waviness at the edges; the amplitude range should span from that up to clear
crumpling.

Validation: sweep the amplitude and confirm `glcm_contrast` and
`lum_spatial_std_mid` rise while `lum_mean` stays stable (the Lambertian shading
must be normalised to preserve the mean).

Produce a plan before writing any code.
```

---

# Estágio C — Óptica

## 4. `blur_sigma`

**Faixa sugerida:** 0 – 3,0 px (na resolução de render)
**Atributos que move:** `sharp_laplacian_var`, `sharp_tenengrad`, `gradient_mean`, `fft_high_freq_ratio`
**Por que aqui:** desfoque óptico age sobre a cena já formada (papel deformado) e
antes de qualquer coisa fotométrica.

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator with trace, dropouts and crumpling implemented.

Task: add `blur_sigma` (0-3.0 px, default 0), Gaussian blur representing imperfect
camera focus. The name matches the existing field in `Params` in image_pipeline.py.

Requirements:
- Apply IN LINEAR LIGHT. Blurring in sRGB space darkens high-contrast edges, and
  an ECG image is almost entirely black-on-white edges, so the error is visible.
- Kernel size 2*round(3*sigma)+1, odd.
- Apply after crumpling and before any photometric adjustment.
- If rendering is supersampled (see increment 13), sigma is expressed in render
  resolution and must be rescaled if that resolution changes.

Validation: `sharp_laplacian_var` must fall monotonically and steeply (over an
order of magnitude between 0 and 3.0). Confirm `lum_mean` changes by less than 1%.

Produce a plan before writing any code.
```

---

# Estágio D — Iluminação e fotometria

## 5. `illum_strength` / `illum_azimuth_deg`

**Faixa sugerida:** força 0 – 0,5; azimute 0 – 360°
**Atributos que move:** `lum_center_edge_diff`, `lum_spatial_std_low`
**Por que separado da vinheta:** o gradiente é assimétrico (a luz vem de um lado); a
vinheta é radial e simétrica. São causas físicas diferentes — iluminação da cena
versus queda do sistema óptico — e precisam de parâmetros separados.

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator, chain complete up to optical blur.

Task: add non-uniform illumination with `illum_strength` (0-0.5, default 0) and
`illum_azimuth_deg` (0-360). Models window light or an off-axis lamp.

Requirements:
- Multiplicative IN LINEAR LIGHT, before exposure.
- Mask: a linear gradient along the azimuth plus a smooth quadratic term. Do not
  use a step or a purely linear gradient — real falloff from a nearby source is
  approximately quadratic with distance.
- Reuse the azimuth as the horizontal component of the crumple light direction L,
  so crease shadows are consistent with the global illumination.
- NORMALISE the mask to mean 1.0. Without this the gradient competes with the
  exposure parameter and calibration of the two oscillates.
- Calibrate this parameter against `lum_center_edge_diff`, NOT against
  `lum_spatial_std_low`, which is already the vignette target.

Validation: sweep the strength and confirm `lum_center_edge_diff` responds
monotonically and `lum_mean` stays constant within 1%.

Produce a plan before writing any code.
```

## 6. `exposure`

**Faixa sugerida:** 0,3 – 2,5 (ganho multiplicativo)
**Atributos que move:** `lum_mean`, `lab_L_mean`, `value_mean`, `lum_median`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator with non-uniform illumination implemented.

Task: add `exposure` (multiplicative gain in linear light, 0.3-2.5, default 1.0).
The name matches the existing field in `Params` in image_pipeline.py.

Requirements:
- Multiplicative gain IN LINEAR LIGHT, applied after the illumination mask. Gain
  applied in sRGB space corresponds to no physical camera operation.
- No clipping at this stage: stay in float and leave clipping to increment 10,
  otherwise `clip_highlights_pct` saturates and cannot be calibrated.
- Register the parameter in the PAIRS table in image_pipeline.py, paired with
  `lum_mean`, bracket (0.05, 20.0), increasing relation.

Validation: the exposure -> `lum_mean` relation must be monotonic across the whole
bracket, and bisection calibration must close with relative error below 1%.

Produce a plan before writing any code.
```

## 7. `wb_r` / `wb_b`

**Faixa sugerida:** 0,85 – 1,20 cada
**Atributos que move:** `color_cast`, `lab_a_mean`, `lab_b_mean`, `wb_r_over_g`, `wb_b_over_g`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator, exposure implemented.

Task: add white balance gains `wb_r` and `wb_b` (0.85-1.20, default 1.0 each),
representing illuminant colour temperature. Names match the existing fields in
`Params` in image_pipeline.py.

Requirements:
- Per-channel gains in linear light, applied alongside exposure.
- Domain caveat: ECG paper has a RED/PINK grid on white. A white balance shift
  changes both the paper white and the apparent saturation of the grid. Check
  visually that the grid stays plausible at the range limits — if it turns strong
  orange or hot pink, tighten the bounds.
- Sample `wb_r` and `wb_b` in a CORRELATED way, not independently: real
  illuminants lie on a curve (warm = high r / low b, cool = the reverse). Use the
  (wb_r_over_g, wb_b_over_g) pairs measured from the real dataset as reference.

Validation: `color_cast` must respond to distance from (1.0, 1.0), and the a*/b*
cloud of generated images must overlap the real dataset cloud in the notebook's
section 5 plot.

Produce a plan before writing any code.
```

## 8. `vignette`

**Faixa sugerida:** 0 – 0,6
**Atributos que move:** `lum_spatial_std_low`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator, illumination and white balance photometry done.

Task: add `vignette` (0-0.6, default 0), radial luminance falloff of the optical
system. The name matches the existing field in `Params` in image_pipeline.py.

Requirements:
- Radial mask (1 - v*r^2) with r normalised by half the diagonal.
- MANDATORY: normalise the mask to mean 1.0 before multiplying. Without this the
  vignette competes directly with exposure. In an earlier version of this solver
  that single omission drove calibration to 85% median error through oscillation
  between the two parameters.
- Apply in linear light, after white balance.
- Register in PAIRS paired with `lum_spatial_std_low`, bracket (0.0, 0.9),
  increasing.

Validation: with `illum_strength=0` and `crumple_amplitude=0`, sweep the vignette
and confirm a monotonic response in `lum_spatial_std_low` with `lum_mean` stable
within 1%. Then repeat with crumpling and illumination gradient active and confirm
that joint calibration still converges in 4 sweeps.

Produce a plan before writing any code.
```

## 9. `contrast`

**Faixa sugerida:** 0,6 – 1,8
**Atributos que move:** `contrast_rms`, `dynamic_range`, `lum_iqr`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator, linear-light stage of the chain complete.

Task: add `contrast` (0.6-1.8, default 1.0), a tone curve applied in DISPLAY SPACE
(after the linear->sRGB conversion), which is where the notebook metrics are
measured. The name matches the existing field in `Params` in image_pipeline.py.

Requirements:
- Curve anchored at mid grey: out = 0.5 + (in - 0.5) * contrast. Anchoring
  anywhere else shifts `lum_mean` and breaks the already-calibrated exposure.
- Apply after the sRGB conversion, before clipping.
- Register in PAIRS paired with `contrast_rms`, bracket (0.1, 4.0), increasing.

Validation: sweep contrast and confirm `contrast_rms` responds monotonically while
`lum_mean` varies less than 2%. Re-calibrate exposure and contrast jointly and
verify both close below 1% error.

Produce a plan before writing any code.
```

## 10. `black_point` / `white_point`

**Faixa sugerida:** preto 0 – 0,10; branco 0,90 – 1,0
**Atributos que move:** `clip_shadows_pct`, `clip_highlights_pct`, `lum_p05`, `lum_p95`, `dark_pixel_pct`, `bright_pixel_pct`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator with the contrast curve implemented.

Task: add `black_point` (0-0.10, default 0) and `white_point` (0.90-1.0, default
1.0), shadow and highlight clipping. Names match the existing fields in `Params`
in image_pipeline.py.

Requirements:
- out = clip((in - black_point) / (white_point - black_point), 0, 1), applied
  after contrast.
- Guard against division by zero if the two points approach each other.
- Domain relevance: photos of white paper under strong light frequently blow out
  the paper white. This parameter reproduces that, and is likely one of the most
  important for matching real photos.
- Calibrate `white_point` against `clip_highlights_pct` and `black_point` against
  `clip_shadows_pct`; they are effectively decoupled from each other.

Validation: confirm `clip_highlights_pct` responds to the white point and that
`lum_p95` follows. Compare the generated `clip_highlights_pct` ECDF against the
real one.

Produce a plan before writing any code.
```

## 11. `saturation`

**Faixa sugerida:** 0,5 – 1,5
**Atributos que move:** `saturation_mean`, `colorfulness`, `chroma_mean`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator, tone chain complete.

Task: add `saturation` (0.5-1.5, default 1.0), chroma scaling. The name matches
the existing field in `Params` in image_pipeline.py.

Requirements:
- Scale a* and b* in CIELAB, not S in HSV. Scaling in HSV distorts perceived
  lightness and, in an ECG image, changes the paper white along with the grid.
- Narrower range than in a generic generator: the red grid saturation is the
  dominant colour information in the image, and extreme values look obviously
  fake. Verify visually before widening.
- Register in PAIRS paired with `colorfulness`, bracket (0.0, 4.0), increasing.

Validation: sweep and confirm monotonicity in `colorfulness` and
`saturation_mean` with `lab_L_mean` stable.

Produce a plan before writing any code.
```

## 12. `hue_rotation`

**Faixa sugerida:** −8° a +8° (ATENÇÃO: muito mais estreita que num gerador genérico)
**Atributos que move:** `hue_mean_deg`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator with saturation implemented.

Task: add `hue_rotation` (-8 to +8 degrees, default 0), hue rotation in the a*/b*
plane of CIELAB. The name matches the existing field in `Params` in
image_pipeline.py.

Critical domain requirement: the ECG paper grid is red by standardisation. A wide
hue rotation produces a green or blue grid, which does not exist. This parameter
only covers the residual colour-temperature variation that white balance did not
capture — keep the range narrow and validate the limits visually.

Before implementing, measure `hue_mean_deg` on the real dataset and size the range
from the observed standard deviation, not from the figure suggested above.

Validation: confirm `hue_mean_deg` tracks the rotation and that the grid stays
unambiguously red/pink at +/-8 degrees.

Produce a plan before writing any code.
```

---

# Estágio E — Sensor e codificação

## 13. `output_width` / `output_height` / `supersample`

**Faixa sugerida:** definida pelos pares (width, height) reais do dataset
**Atributos que move:** `width`, `height`, `aspect_ratio`, `megapixels`

> **Correção de ordem em relação ao `PARAMETROS_GERADOR.md`:** naquele documento
> width e height aparecem primeiro. Para ECG a posição correta é aqui, imediatamente
> antes do ruído. O motivo é o traço: linhas de 0,3 mm e grade milimetrada produzem
> aliasing severo se renderizadas direto na resolução final. Renderizar 2–3× maior
> e reduzir aqui dá anti-aliasing correto e é o que a câmera fisicamente faz.

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator with the full photometric chain in place, rendering at a
fixed resolution.

Task: introduce supersampling. Render the whole chain up to this point at a
`supersample` factor (2 or 3) above the output resolution, and downscale here to
(`output_width`, `output_height`) with INTER_AREA.

Requirements:
- Sample (output_width, output_height) pairs jointly from the real dataset, never
  marginally: independent width and height produce aspect ratios no camera makes.
- Every parameter expressed in pixels (blur_sigma, crumple displacement) must be
  scaled by the supersample factor. Parameters in millimetres do not need it —
  one more reason increment 1 used mm.
- The downscale happens AFTER all photometry and BEFORE noise: sensor noise
  originates at sensor resolution and must not be downscaled with the image.

Validation: compare `sharp_laplacian_var` and `edge_density` before and after
introducing supersampling at the same output resolution. They must RISE (less
aliasing, cleaner edges). If they fall, the downscale is in the wrong position.

Produce a plan before writing any code.
```

## 14. `sensor_noise`

**Faixa sugerida:** 0 – 8 (níveis 0–255)
**Atributos que move:** `noise_sigma`, `noise_residual`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator with supersampling and downscaling to output resolution.

Task: add `sensor_noise` (0-8, default 0), sensor noise. The name matches the
existing field in `Params` in image_pipeline.py — note it is deliberately NOT
called `noise_sigma`, which is the name of the measured feature it targets.

Requirements:
- SIGNAL-DEPENDENT noise, not homoscedastic Gaussian:
  effective_sigma = sensor_noise * sqrt(clip(pixel/255, 0.02, 1.0)).
  Real sensors have a photon component proportional to the square root of signal.
  This matters a lot for ECG images, which have large white paper areas (much
  noise) and a thin black trace (little noise); uniform noise looks visibly wrong
  in both regions.
- Apply at output resolution, after the downscale.
- Register in PAIRS paired with `noise_sigma`, bracket (0.0, 40.0), increasing.

KNOWN LIMIT: this parameter only ADDS noise. If the rendered content already has a
noise floor above the target, the target is unreachable — in the previous
generator test this happened in 6 of 12 targets. Confirm the base render is clean
and that `noise_sigma` of the raw content sits below the P1 of the target
distribution. If it does not, the problem is in the renderer, not in this
parameter.

Validation: sweep and confirm monotonicity; verify the solver reports the target
as unreachable instead of saturating when the target sits below the floor.

Produce a plan before writing any code.
```

## 15. `jpeg_q`

**Faixa sugerida:** 55 – 98
**Atributos que move:** `blockiness`, `bits_per_pixel`, `file_size_kb`

```
LANGUAGE RULE: all code in English — parameter names, variables, functions,
classes, comments, docstrings, log and error messages, file names, test names.
No Portuguese identifiers or comments anywhere in the codebase.

Context: ECG generator with the full chain in place.

Task: add `jpeg_q` (55-98, default 95), JPEG compression quality. The name matches
the existing field in `Params` in image_pipeline.py.

Requirements:
- Encode AND decode back, so the artefacts enter the in-memory image. Saving at
  low quality is not enough if validation measures the image before saving.
- Last stage of the chain, no exceptions.
- Domain relevance: aggressive JPEG creates mosquito noise around thin
  high-contrast lines, exactly the ECG trace case. It is a very characteristic
  artefact of real photos shared over messaging apps or email — check whether the
  real dataset has that provenance. If it does, the low range (55-75) matters more
  than the high one.

Validation: `blockiness` and `bits_per_pixel` must respond monotonically. Compare
the generated `bits_per_pixel` distribution against the real one: it is one of the
easiest features to match and one of the most revealing when a generator is badly
calibrated.

Produce a plan before writing any code.
```

---

# Ajustes na medição (já implementados)

`measure()` em `image_pipeline.py` já devolve as três escalas espaciais, resolvendo
a disputa entre amassado, iluminação e vinheta:

```python
out["lum_spatial_std"]      = _spatial_std(Y, 4)    # notebook compatibility
out["lum_spatial_std_low"]  = _spatial_std(Y, 3)    # illumination, vignette
out["lum_spatial_std_mid"]  = _spatial_std(Y, 12)   # creases, surface texture
out["lum_center_edge_diff"] = ...
```

Replique as duas colunas novas no extrator do notebook para que a comparação
real vs. gerado use as mesmas métricas.

Vale também acrescentar duas métricas específicas de ECG para validação — elas não
são parâmetros, são checagens de que o gerador não quebrou a estrutura:

- **`grid_periodicity`**: pico da FFT 2D nas frequências correspondentes a 1 mm e
  5 mm. Se o amassado ou o desfoque destruírem esse pico, a imagem deixou de
  parecer papel milimetrado.
- **`trace_continuity`**: fração do maior componente conexo escuro. Cai com
  `trace_dropout_rate` de forma previsível; queda inesperada indica bug.

---

# Parâmetros que ficaram de fora

Estes não estavam no seu pedido, mas a foto de exemplo os exibe claramente e, sem
eles, o gerador não cobre a distribuição real de fotos de ECG:

- **`perspective_*`** — a folha do exemplo está levemente rotacionada e em trapézio.
  Entra no estágio B, logo depois do amassado. É provavelmente o maior gap.
- **`background_*`** — mesa de madeira visível, pilha de folhas por baixo, sombra
  do papel sobre a mesa. Afeta fortemente `lum_mean`, `colorfulness` e todos os
  atributos de cor, porque ocupa área relevante do quadro.
- **`drop_shadow_*`** — a borda esquerda da folha no exemplo tem sombra suave.

Se o modelo final for treinado para ler ECG de fotos reais, esses três valem mais
que metade da lista acima. Sugiro implementá-los logo depois do incremento 3.
