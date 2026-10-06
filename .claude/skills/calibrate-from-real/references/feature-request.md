# Feature request for a gap the generator cannot draw

One file per gap: `<run_dir>/feature_requests/<short-name>.md`. The skill writes the
request; it does not implement it. Implementation is a separate session in the generator
repository, started by the user.

A request earns its place with evidence from THIS lot: how often, on which pages, and what
the detector does there. "Real ECGs sometimes have moire" is not a request.

## Template

```markdown
# Feature request: <what the page has that the generator cannot draw>

## Evidence
- Lot: <lot dir>, <n> usable pages (<date>).
- Frequency: <k> of <n> control pages (<share>%, 90% interval <low>-<high>%); <k> of <n>
  among the worst pages by proxy.
- Example pages (local views, not for publishing): <run_dir>/audit/<slug>/ ...
- Detector on those pages: proxy <values>, leads kept <values>, what goes wrong
  (names swapped, leads missed, boxes merged) if visible.
- Population: <name>; goes together with <other traits of that population>.

## What the real pages look like
<Three to six sentences a person who has not seen the pages could draw from: geometry in
mm where it was measured, colours as RGB, which printed elements move or change.>

## What exists today
<The nearest option and why it does not produce this: file and key.>

## Proposed behaviour
<What a new option would draw. Its parameters and their ranges, each with the measurement
it comes from. What it must leave untouched (lead boxes, the JSON contract nb 4.3 reads).>

## Draft prompt for the implementation session
<Self-contained: the generator repo path and branch (upstream-baseline, Python 3.10 venv,
English only), the files likely involved, the behaviour above, and these standing rules:

- a new feature is a realism group of its own: p per record, its own RNG stream, appended
  at the END of GROUPS in realism.py - never a new key under randomize:, which would
  re-roll every key sorting after it;
- its draw is written to the page JSON under realism: {...}, with the realised parameters;
- with p 0 the pages are byte-identical to today's;
- a one-group experiment variant extends the recipe, like batch_ptbxl_inc_exp_*.yaml;
- verified by a smoke render looked at beside the real example pages.>

## Priority
<high / medium / low, from frequency in the control sample x how badly the detector does
on those pages. State both numbers. A trait seen once among the worst pages and never in
the control sample is "low, watch for it in the next lot".>
```
