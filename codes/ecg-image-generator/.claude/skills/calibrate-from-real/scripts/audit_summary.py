#!/usr/bin/env python
"""Steps 2 and 4 - validate the visual audit and count what it found.

    audit_summary.py <run_dir> [--worst-n 12] [--partial]

Reads <run_dir>/audit.jsonl (one JSON object per audited page, vocabulary in
assets/audit_schema.json) and <run_dir>/audit/selection.csv. Refuses unknown values, so
the counts mean the same thing from one lot to the next. Writes <run_dir>/audit_summary.md:

  - every field's values, counted in the CONTROL sample with a 90% Wilson interval - that
    is the estimate of how common a trait is in the lot;
  - the same values among the WORST pages by proxy (the `worst-n` lowest ranks audited,
    whatever their group - so a lot audited whole still has its worst pages) - a trait far
    more common among the worst pages than in the control is a candidate gap;
  - `row_pitch_mm` and the other optional numbers: p10 / median / p90;
  - the cost of each trait to the detector: median proxy and mean leads kept on the pages
    that show it against the pages that do not (from ranking.csv) - the "how badly" half
    of a gap's priority;
  - the populations named in the audit, with their control share;
  - candidate same-exam duplicates: pages sharing an exam_stamp (record the confirmed
    ones with `intake.py same-exam` and run this again: they then count once).
"""
import argparse
import collections
import json
import math
from pathlib import Path

from _common import read_csv

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / "assets" / "audit_schema.json").read_text())


def wilson(k, n, z=1.645):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


parser = argparse.ArgumentParser()
parser.add_argument("run_dir")
parser.add_argument("--worst-n", type=int, default=None,
                    help="how many lowest-proxy pages count as 'worst' (default: 12, or a fifth of a small lot)")
parser.add_argument("--partial", action="store_true", help="summarise an audit still in progress")
args = parser.parse_args()
run_dir = Path(args.run_dir)
selection = {row["slug"]: row for row in read_csv(run_dir / "audit" / "selection.csv")}
if args.worst_n is None:
    args.worst_n = min(12, max(3, round(0.2 * len(selection))))
pages, errors = [], []
for n, line in enumerate((run_dir / "audit.jsonl").read_text().splitlines(), 1):
    if not line.strip():
        continue
    page = json.loads(line)
    where = "line %d (%s)" % (n, page.get("slug"))
    for key in SCHEMA["required"]:
        if key not in page:
            errors.append("%s: missing %s" % (where, key))
    if page.get("slug") not in selection:
        errors.append("%s: slug not in audit/selection.csv" % where)
    elif page.get("group") != selection[page["slug"]]["group"]:
        errors.append("%s: group %r, selection.csv says %r" % (where, page.get("group"), selection[page["slug"]]["group"]))
    for key, allowed in SCHEMA["single"].items():
        if key in page and page[key] not in allowed:
            errors.append("%s: %s=%r not in %s" % (where, key, page[key], allowed))
    for key in SCHEMA.get("numeric_optional", []):
        value = page.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            errors.append("%s: %s=%r must be a number or null" % (where, key, value))
    for key, allowed in SCHEMA["multi"].items():
        values = page.get(key, [])
        if not isinstance(values, list) or set(values) - set(allowed):
            errors.append("%s: %s=%r must be a list drawn from %s" % (where, key, values, allowed))
    pages.append(page)
missing = sorted(set(selection) - {p.get("slug") for p in pages})
if missing and not args.partial:
    errors.append("selected but not audited: %s (use --partial while the audit is in progress)" % ", ".join(missing))
if errors:
    raise SystemExit("audit.jsonl has %d problem(s):\n  %s" % (len(errors), "\n  ".join(errors)))

# an exam counts once: pages recorded with `intake.py same-exam` after the audit are left out
manifest = {row["slug"]: row for row in read_csv(run_dir / "manifest.csv")}
repeats = [p for p in pages if manifest.get(p["slug"], {}).get("same_exam_as")]
pages = [p for p in pages if p not in repeats]
control = [p for p in pages if p["group"] == "control"]
worst = [p for p in pages if int(selection[p["slug"]]["rank"]) < args.worst_n]
out = ["# Audit summary" + (" (PARTIAL: %d selected page(s) not audited yet)" % len(missing) if missing else ""), "",
       "%d page(s) audited: %d control (random, or the whole lot; frequencies come from here), %d among the %d "
       "worst by proxy (discovery; they may also be control pages)." % (len(pages), len(control), len(worst),
                                                                     args.worst_n), ""]


def tally(group, key, multi):
    counter = collections.Counter()
    for page in group:
        counter.update(page.get(key, []) if multi else [page.get(key, "")])
    return counter


ranking = {row["slug"]: row for row in read_csv(run_dir / "ranking.csv")}


def cost(group, has):
    """median proxy / mean leads of the pages with the trait, and of the others."""
    def stats(pages):
        proxies = sorted(float(ranking[p["slug"]]["proxy"]) for p in pages if p["slug"] in ranking)
        leads = [int(ranking[p["slug"]]["n_leads"]) for p in pages if p["slug"] in ranking]
        if not proxies:
            return "-"
        return "%.2f / %.1f" % (proxies[len(proxies) // 2], sum(leads) / len(leads))
    return stats([p for p in group if has(p)]), stats([p for p in group if not has(p)])


for key in list(SCHEMA["single"]) + list(SCHEMA["multi"]) + ["population", "printer_format"]:
    multi = key in SCHEMA["multi"]
    in_control, in_worst = tally(control, key, multi), tally(worst, key, multi)
    # every value of the vocabulary gets a row, so an absent trait shows as 0/n, not as nothing
    for value in SCHEMA["single"].get(key, []) + SCHEMA["multi"].get(key, []):
        in_control.setdefault(value, 0)
    out += ["## %s" % key, "", "| value | control | control share (90% CI) | worst | worst share | "
            "proxy / leads with | proxy / leads without | example pages |",
            "|---|---|---|---|---|---|---|---|"]
    for value in sorted(set(in_control) | set(in_worst), key=lambda v: -(in_control[v] + in_worst[v])):
        low, high = wilson(in_control[value], len(control))
        share = "%.0f%% (%.0f-%.0f%%)" % (100 * in_control[value] / max(len(control), 1), 100 * low, 100 * high)
        examples = list(dict.fromkeys(p["slug"] for p in worst + control
                                      if (value in p.get(key, []) if multi else p.get(key, "") == value)))[:4]
        with_, without = cost(control, (lambda p, v=value: v in p.get(key, [])) if multi else (lambda p, v=value: p.get(key, "") == v))
        out.append("| %s | %d/%d | %s | %d/%d | %s | %s | %s | %s |" % (
            value, in_control[value], len(control), share, in_worst[value], len(worst),
            ("%.0f%%" % (100 * in_worst[value] / len(worst))) if worst else "-", with_, without, ", ".join(examples)))
    out.append("")

for key in SCHEMA.get("numeric_optional", []):
    values = sorted(float(p[key]) for p in control if isinstance(p.get(key), (int, float)) and not isinstance(p.get(key), bool))
    out += ["## %s (control pages)" % key, ""]
    if values:
        pick = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]
        out += ["n=%d: p10 %.1f, median %.1f, p90 %.1f" % (len(values), pick(0.1), pick(0.5), pick(0.9)), ""]
    else:
        out += ["not recorded", ""]

stamps = collections.defaultdict(list)
for page in pages:
    if page.get("exam_stamp"):
        stamps[page["exam_stamp"].strip().lower()].append(page["slug"])
shared = {stamp: slugs for stamp, slugs in stamps.items() if len(slugs) > 1}
out += ["## Same-exam candidates (shared exam_stamp)", ""]
out += ["- `%s`: %s" % (stamp, ", ".join(slugs)) for stamp, slugs in sorted(shared.items())] or ["none among the audited pages"]
out += ["", "## Repeats of an exam, left out of the counts above", ""]
slug_of = {row["file"]: row["slug"] for row in manifest.values()}
out += ["- %s repeats %s" % (p["slug"], slug_of.get(manifest[p["slug"]]["same_exam_as"], manifest[p["slug"]]["same_exam_as"]))
        for p in repeats] or ["none recorded"]
out += ["", "## Pages not judged in distribution (`partly` or `no`)", ""]
out += ["- %s (%s, proxy %s): %s" % (p["slug"], p["group"], selection[p["slug"]]["proxy"], p["notes"])
        for p in pages if p["in_distribution"] != "yes"] or ["none"]

(run_dir / "audit_summary.md").write_text("\n".join(out) + "\n")
print("\n".join(out))
print("\nwrote %s/audit_summary.md" % run_dir)
