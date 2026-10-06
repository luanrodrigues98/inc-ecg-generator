#!/usr/bin/env python
"""Step 5/6 - check a batch YAML BEFORE anything renders. Runs in the GENERATOR's venv:

    cd <generator root>
    .venv310/bin/python .claude/skills/calibrate-from-real/scripts/preflight.py <batch.yaml> --probe
    .venv310/bin/python .claude/skills/calibrate-from-real/scripts/preflight.py <batch.yaml> --full

It loads the YAML through the runner's own load_config + build_args, so an unknown key or
a bad realism block fails here and not twenty minutes into a run, then checks:

  - the YAML is not a file git already tracks (a calibration never edits an existing recipe);
  - the output directory does not exist or is empty - the runner writes flat, so a second
    run into a directory interleaves two datasets under one name (ptbxl_synthetic_4000 was
    overwritten that way once);
  - enough RAM: the peak grows with dpi^2 (21.6 GB measured at 598 dpi, supersample 2, one
    worker) and a run that reaches swap dies. A refusal for a probe; for a full lot, which
    legitimately runs near the machine's limit, a warning to pass on to the user;
  - enough disk: a page is 1.764e-4 * dpi^2 + 7.2 MB;
  - no other render is running.

--probe additionally enforces the limits of a probe lot the skill may render by itself:
at most 24 pages for the whole calibration run (the probe_* directories already in the
run count), one worker, at most 450 dpi. --full enforces nothing more and renders
nothing: it prints the cost and the command for the USER to start.

Exit status 0 means "safe to run"; anything else, do not render.
"""
import argparse
import os
import shutil
import subprocess
import sys

GENERATOR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__))))))
RUN_LOG_DIR = os.path.expanduser("~/.cache/inc-ecg-generator-run")

parser = argparse.ArgumentParser()
parser.add_argument("config")
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--probe", action="store_true")
mode.add_argument("--full", action="store_true")
args = parser.parse_args()
config_path = os.path.abspath(args.config)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # the runner imports TensorFlow: keep its banner out
os.chdir(GENERATOR)
sys.path.insert(0, GENERATOR)
sys.argv = [sys.argv[0]]
import run_batch_from_config as runner  # noqa: E402

problems = []

tracked = subprocess.run(["git", "-C", GENERATOR, "ls-files", "--error-unmatch", config_path],
                         capture_output=True).returncode == 0
if tracked:
    problems.append("%s is tracked by git: write a NEW YAML (extends: the base) instead of editing a recipe" % config_path)

config = runner.load_config(config_path)
batch_args, randomize = runner.build_args(config, config_path)
output = os.path.normpath(os.path.join(GENERATOR, batch_args.output_directory))
if os.path.isdir(output) and os.listdir(output):
    problems.append("output_directory %s already holds files: choose a new directory" % output)

pages = batch_args.max_num_images
workers = int(batch_args.max_workers or 1)
spec = randomize.get("resolution")
if spec is None:
    dpi_low = dpi_high = int(batch_args.resolution)
else:
    values = next(iter(spec.values()))
    dpi_low, dpi_high = int(min(values)), int(max(values))
supersample = float(getattr(batch_args, "supersample", 1) or 1)

peak_gb = 21.6 * (dpi_high / 598.0) ** 2 * (supersample / 2.0) ** 2 * workers
available_gb = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable")) / 2 ** 20
warnings = []
if available_gb < peak_gb + 2:
    message = ("RAM: peak about %.1f GB (dpi %d, supersample %g, %d worker(s)), %.1f GB available"
               % (peak_gb, dpi_high, supersample, workers, available_gb))
    if args.probe:
        problems.append(message + " - lower the resolution or free memory")
    else:
        # the v2 batch ran at this peak with one worker on this machine: tight, not impossible
        warnings.append(message + " - close every other heavy job before starting, and keep max_workers at 1")

dpi_mean_sq = (dpi_low ** 2 + dpi_low * dpi_high + dpi_high ** 2) / 3.0
count = pages if pages and pages > 0 else None
parent = output
while not os.path.isdir(parent):
    parent = os.path.dirname(parent)
free_gb = shutil.disk_usage(parent).free / 2 ** 30
if count is None:
    problems.append("max_num_images is unbounded: set it")
else:
    disk_gb = count * (1.764e-4 * dpi_mean_sq + 7.2) / 1024
    hours = count * 1.66e-4 * dpi_mean_sq / 3600 / max(1, min(workers, 4) * 0.6)
    if free_gb < 1.2 * disk_gb + 2:
        problems.append("disk: about %.1f GB to write, %.1f GB free under %s" % (disk_gb, free_gb, parent))

others = []
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        argv = open("/proc/%s/cmdline" % pid, "rb").read().decode(errors="replace").split("\0")
    except OSError:
        continue
    # a python process whose script IS the runner - not a shell whose command line mentions it
    if os.path.basename(argv[0]).startswith("python") and any(a.endswith("run_batch_from_config.py") for a in argv[1:3]):
        others.append("%s %s" % (pid, " ".join(argv[:4])))
if others:
    problems.append("another render is running (one at a time - RAM):\n      %s" % "\n      ".join(others))

if args.probe:
    # the 24-page budget is per calibration run: count what earlier probes of this run rendered
    run_dir = os.path.dirname(output)
    done = sum(len([f for f in os.listdir(os.path.join(run_dir, d)) if f.endswith(".png")])
               for d in os.listdir(run_dir) if d.startswith("probe_") and os.path.isdir(os.path.join(run_dir, d))) \
        if os.path.isdir(run_dir) else 0
    if count is None or count + done > 24:
        problems.append("probe: at most 24 pages per calibration run - %d already rendered under %s, this one asks "
                        "for %s" % (done, run_dir, pages))
    if workers != 1:
        problems.append("probe: max_workers must be 1 (got %d)" % workers)
    if dpi_high > 450:
        problems.append("probe: resolution must stay <= 450 dpi (got up to %d) - pin it with "
                        "randomize: {resolution: {randint: [400, 400]}}" % dpi_high)

print("config           : %s" % config_path)
print("output_directory : %s" % output)
print("pages            : %s   workers %d   dpi %d-%d   supersample %g" % (pages, workers, dpi_low, dpi_high, supersample))
realism = getattr(batch_args, "realism", None)
if isinstance(realism, dict):
    print("realism (p)      : %s" % ", ".join(
        "%s %s" % (group, entry.get("p") if isinstance(entry, dict) else entry) for group, entry in realism.items()))
else:
    print("realism          : %r  <- NO realism block: every group off, the old generator's pages" % (realism,))
if count is not None:
    print("estimated cost   : %.1f GB on disk (%.0f GB free), peak RAM %.1f GB (%.1f GB available), at most %s" % (
        disk_gb, free_gb, peak_gb, available_gb,
        ("%.0f min" % (60 * hours)) if hours < 2 else ("%.0f h" % hours)))
if problems:
    print("\nNOT SAFE TO RUN:")
    for problem in problems:
        print("  - %s" % problem)
    sys.exit(1)

for warning in warnings:
    print("WARNING: %s" % warning)
relative = os.path.relpath(config_path, GENERATOR)
if relative.startswith(".."):
    relative = config_path
if args.probe:
    print("\nOK. Probe command (the skill may run this):")
    print("  cd %s && ./.venv310/bin/python run_batch_from_config.py %s" % (GENERATOR, relative))
else:
    name = os.path.splitext(os.path.basename(config_path))[0]
    print("\nOK. FULL LOT - for the user to start, not the skill:")
    print("  cd %s && nohup ./.venv310/bin/python run_batch_from_config.py %s > %s/%s.log 2>&1 &" % (
        GENERATOR, relative, RUN_LOG_DIR, name))
    print("  tail -f %s/%s.log     # the run resumes with skip_existing if interrupted" % (RUN_LOG_DIR, name))
