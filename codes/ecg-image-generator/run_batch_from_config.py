#!/usr/bin/env python
#Run a batch of ECG images from a YAML configuration file.
#
#The batch driver is driven entirely by command line flags: its --config_file YAML
#(config.yaml) carries only the plotting constants handed to get_paper_ecg, and never
#reaches the generation parameters, which are read off the argparse namespace. This
#script is the missing consumer: it fills that namespace from a YAML file so a run is
#described by one reviewable, version-controllable document instead of a long command.
#
#It adds two things the batch driver cannot express:
#
#  - a `randomize:` block, which draws a parameter PER RECORD instead of fixing it for
#    the whole batch. The driver samples crumple, blur, noise and rotation per image
#    already, but layout, trace geometry and resolution are single values for the run,
#    which makes 3000 images that differ only in their distortion. num_columns in
#    particular has no random mode at all.
#  - seeding of every RNG in the chain. run() seeds `random` only; numpy and imgaug are
#    left on their global state, so --augment and any fractional bernoulli probability
#    make a batch that cannot be reproduced from its seed.
#
#Usage:
#    .venv310/bin/python run_batch_from_config.py batch_ptbxl_3000.yaml
import os, sys, random, shutil, tempfile, yaml
import numpy as np
from tqdm import tqdm

#The config path is an argument to THIS script, so it is resolved against the shell's
#directory, before the chdir moves it. Paths inside the YAML keep the tool's own
#convention and resolve against the generator root.
INVOCATION_CWD = os.getcwd()
GENERATOR_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(GENERATOR_ROOT)

import imgaug
from helper_functions import find_records
from gen_ecg_images_from_data_batch import get_parser
from gen_ecg_image_from_data import run_single_file

REQUIRED_KEYS = ('input_directory', 'output_directory')
DRAWS = ('choice', 'uniform', 'randint')


def draw(spec, rng):
    """Draw one value from a randomize entry: choice, uniform or randint."""
    kind = next(iter(spec))
    if kind == 'choice':
        return rng.choice(spec[kind])
    low, high = spec[kind]
    return rng.uniform(low, high) if kind == 'uniform' else rng.randint(low, high)


def validate_randomize(randomize, known, static_keys, config_path):
    for key, spec in randomize.items():
        if key not in known:
            raise SystemExit("%s: randomize: unknown parameter %r" % (config_path, key))
        if key in static_keys:
            raise SystemExit(
                "%s: %r is set both as a fixed value and under randomize:. "
                "Remove one - the fixed value would be overwritten every record."
                % (config_path, key))
        if not isinstance(spec, dict) or len(spec) != 1 or next(iter(spec)) not in DRAWS:
            raise SystemExit(
                "%s: randomize: %s must be a single %s entry, e.g. {uniform: [0.2, 0.6]}"
                % (config_path, key, '/'.join(DRAWS)))
        kind = next(iter(spec))
        if kind != 'choice':
            bounds = spec[kind]
            if not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] > bounds[1]:
                raise SystemExit(
                    "%s: randomize: %s.%s must be [low, high] with low <= high"
                    % (config_path, key, kind))
        elif not spec[kind]:
            raise SystemExit("%s: randomize: %s.choice is empty" % (config_path, key))


def build_args(config, config_path):
    """Overlay a config dict onto the batch parser's own namespace.

    Starting from the parser guarantees every attribute run_single_file touches exists
    and carries the tool's own default, so a key left out of the YAML behaves exactly as
    it would if the flag were left off the command line.
    """
    args = get_parser().parse_args(['-i', '', '-o', ''])
    known = set(vars(args))
    randomize = config.pop('randomize', None) or {}

    #Reject unknown keys rather than ignoring them. A parameter that parses and does
    #nothing is the failure mode this codebase already suffers from with
    #--deterministic_rot; a config file silently dropping vignette: 0.3 because the
    #chain has not reached that increment yet would be the same trap with a new face.
    unknown = sorted(set(config) - known)
    if unknown:
        raise SystemExit(
            "%s: unknown parameter(s): %s\n"
            "Keys are argparse destination names, not flag spellings "
            "(--hws is handwriting_size_factor, -rot is rotate, -ca is crease_angle)."
            % (config_path, ', '.join(unknown)))

    validate_randomize(randomize, known, set(config), config_path)

    missing = [key for key in REQUIRED_KEYS if not config.get(key)]
    if missing:
        raise SystemExit("%s: missing required key(s): %s" % (config_path, ', '.join(missing)))

    for key, value in config.items():
        setattr(args, key, value)

    #get_augment reads json_dict['leads'] unconditionally, but run_single_file sets
    #json_dict to None when store_config is 0, so the pair crashes with a TypeError
    #several minutes into a long batch. Refuse it up front instead.
    if args.augment and not args.store_config:
        raise SystemExit(
            "%s: augment requires store_config 1 or 2 "
            "(get_augment reads the annotation dict that store_config 0 leaves as None)."
            % config_path)

    #The augment stage draws its noise with random.choice(range(1, noise + 1)), which
    #raises IndexError on the empty range that noise 0 produces. The render is written
    #before that point, so the batch dies leaving a directory of undistorted PNGs.
    if args.augment and not args.deterministic_noise and args.noise < 1:
        raise SystemExit(
            "%s: noise must be at least 1 while augment is on "
            "(the per-image draw is over range(1, noise + 1), empty at 0). "
            "Use augment: false to switch the stage off." % config_path)

    #exposure is the one parameter with two randomisation paths, because its neutral is a
    #gain of 1.0 rather than 0 and its jitter therefore could not be folded into the value
    #the way --crumple_amplitude and --blur_sigma fold theirs. validate_randomize catches
    #a key that is fixed AND randomized, but this pair is a key that is randomized here and
    #jittered again inside run_single_file, which is a second draw on top of the first and
    #widens the distribution past whatever range was declared.
    if 'exposure' in randomize and args.exposure_jitter_stops > 0:
        raise SystemExit(
            "%s: exposure is drawn per record under randomize: and jittered again by "
            "exposure_jitter_stops %s. Use one or the other - the randomize block is the "
            "per-record draw, exposure_jitter_stops is for running the batch driver "
            "without this runner." % (config_path, args.exposure_jitter_stops))

    #A gain of 0 or less is meaningless, and it also reaches json.dumps: run_single_file
    #records log2(exposure) beside the gain, which is -Infinity at 0 and NaN below it.
    #Python writes both without complaint and every strict JSON parser rejects them, so
    #the dataset would fail to load on the annotation rather than on the image.
    if 'exposure' in randomize:
        spec = randomize['exposure']
        kind = next(iter(spec))
        exposure_low = min(spec[kind]) if kind == 'choice' else spec[kind][0]
    else:
        exposure_low = args.exposure
    if exposure_low <= 0:
        raise SystemExit(
            "%s: exposure must be greater than 0, got %s. 1.0 is the neutral gain and 0 "
            "would be a black page; the annotation also records log2(exposure), which is "
            "not representable at 0." % (config_path, exposure_low))

    #The white balance is the one parameter whose per-record draw MUST NOT go through the
    #randomize block, and this guard is the only one here that refuses a key outright
    #rather than refusing a combination. The reason is in the roteiro: wb_r and wb_b have
    #to be sampled in a CORRELATED way, because a real illuminant has one degree of
    #freedom - warm light is high r AND low b - and randomize: draws every key from its own
    #independent uniform, which would produce the (high r, high b) corner, a lamp that is
    #simultaneously warm and cool and does not exist. wb_mired_jitter is the correlated
    #draw, along the daylight locus, and it is inside run_single_file.
    #wb_mired_jitter itself is refused for a different reason: drawing the HALF WIDTH per
    #record and then drawing again inside it is the same double draw the exposure guard
    #below rejects, and it widens the distribution past whatever range was declared.
    white_balance_keys = sorted(set(randomize) & {'wb_r', 'wb_b', 'wb_mired_jitter'})
    if white_balance_keys:
        raise SystemExit(
            "%s: randomize: %s cannot be drawn here. randomize: samples every key "
            "independently, and wb_r/wb_b must move together along the illuminant locus "
            "(warm light is high r AND low b). Set wb_r and wb_b as fixed values and use "
            "wb_mired_jitter for the per-record draw, which is correlated by construction."
            % (config_path, ', '.join(white_balance_keys)))

    #A gain of 0 or less on a colour channel removes that channel from the image, which is
    #not a white balance at any setting. Refused for the same reason as the exposure below.
    for key in ('wb_r', 'wb_b'):
        if getattr(args, key) <= 0:
            raise SystemExit(
                "%s: %s must be greater than 0, got %s. 1.0 is the neutral gain and 0 "
                "would delete the channel." % (config_path, key, getattr(args, key)))

    #deterministic_temp was inert upstream, so its companion --temperature kept a default
    #of 40000 that nothing ever read. Reading the flag makes that default live, and 40000 K
    #is the blue end of imgaug's table - a page the colour of a computer screen. Nobody who
    #switches the flag on means that, so ask for the value explicitly. Use temperature 0 to
    #switch the colour temperature step off entirely, which is what hands ownership of the
    #cast to wb_r and wb_b.
    if args.deterministic_temp and args.temperature == 40000:
        raise SystemExit(
            "%s: deterministic_temp is on but temperature is still the inert upstream "
            "default of 40000, which is the blue extreme of imgaug's table. Set "
            "temperature explicitly - 0 switches the colour temperature step off and "
            "leaves the cast to wb_r/wb_b." % config_path)

    #random_resolution ignores the randomize block and draws from range(50, resolution+1).
    #A 50 dpi ECG is unreadable and the top of that range is enormous, so the two ways of
    #varying resolution must not be mixed.
    if args.random_resolution and 'resolution' in randomize:
        raise SystemExit(
            "%s: random_resolution draws from range(50, resolution + 1) and ignores "
            "randomize: resolution. Set random_resolution: false to use the range you "
            "declared." % config_path)

    return args, randomize


def seed_everything(seed):
    """Seed every RNG the chain draws from.

    run() seeds `random` alone. scipy's bernoulli draws from numpy's global RNG and
    imgaug's noise from its own, so without these two a batch is not reproducible from
    its seed once augment or a fractional probability is in play.
    """
    entropy = abs(int(seed)) % (2 ** 32)
    random.seed(seed)
    np.random.seed(entropy)
    imgaug.seed(entropy)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_batch_from_config.py <config.yaml>")

    config_path = os.path.normpath(os.path.join(INVOCATION_CWD, sys.argv[1]))
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise SystemExit("%s: expected a YAML mapping of parameters" % config_path)

    args, randomize = build_args(config, config_path)

    input_directory = os.path.normpath(os.path.join(os.getcwd(), args.input_directory))
    output_root = os.path.normpath(os.path.join(os.getcwd(), args.output_directory))
    if not os.path.isdir(input_directory):
        raise SystemExit("input_directory does not exist: %s" % input_directory)
    os.makedirs(output_root, exist_ok=True)
    args.input_directory = input_directory

    seed_everything(args.seed)

    print("config           : %s" % config_path)
    print("input_directory  : %s" % input_directory)
    print("output_directory : %s" % output_root)
    print("max_num_images   : %s" % args.max_num_images)
    print("seed             : %s" % args.seed)
    print("randomized       : %s" % (', '.join(sorted(randomize)) or 'none'))
    sys.stdout.flush()

    #find_records mirrors the corpus tree under the directory it is given, as a side
    #effect of listing it. The output here is flat, so that tree goes to a throwaway
    #directory and is dropped.
    scratch = tempfile.mkdtemp(prefix='ecg_find_records_')
    try:
        header_files, recording_files = find_records(input_directory, scratch)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    #Sort what os.walk returned. find_records sorts the files of each directory but never
    #the directory list itself, so the corpus order - and therefore WHICH records a
    #max_num_images cut keeps - is filesystem dependent and not portable. Sorting makes
    #the chosen subset a function of the corpus alone.
    pairs = sorted(zip(header_files, recording_files))

    #A flat output directory makes the record name the only identity a file has, so two
    #records sharing a name would silently overwrite each other. Keep the first of each
    #name. On PTB-XL this drops records100/union, which is a byte-identical copy of
    #01000-02999 and would otherwise put the same signal in the set twice.
    seen = set()
    records = []
    for header_file, recording_file in pairs:
        name = os.path.splitext(os.path.basename(recording_file))[0]
        if name in seen:
            continue
        seen.add(name)
        records.append((header_file, recording_file, name))
    duplicates = len(pairs) - len(records)
    print("records          : %d unique (%d duplicate name(s) skipped)" % (len(records), duplicates))
    sys.stdout.flush()

    #Count the bar in images rather than records: run_single_file returns the number of
    #frames it wrote, and max_num_images caps frames, not records. On this corpus the two
    #coincide at one frame per record, but a longer recording would split into several.
    total = len(records)
    if args.max_num_images != -1:
        total = min(args.max_num_images, total)

    written = 0
    bar = tqdm(total=total, unit='img', desc='rendering', dynamic_ncols=True)
    try:
        for header_file, recording_file, name in records:
            args.input_file = os.path.join(input_directory, recording_file)
            args.header_file = os.path.join(input_directory, header_file)
            args.start_index = -1
            args.output_directory = output_root
            args.encoding = name

            #Key the per-record draws by record name rather than by position in the walk,
            #so a record keeps its characteristics wherever the listing reaches it.
            rng = random.Random('%s:%s' % (args.seed, args.encoding))
            for key, spec in sorted(randomize.items()):
                setattr(args, key, draw(spec, rng))

            bar.set_postfix_str(name, refresh=False)
            written += run_single_file(args)
            #Clamp, so a record yielding more frames than the cap leaves cannot push the
            #bar past its total.
            bar.update(min(written, total) - bar.n)

            if args.max_num_images != -1 and written >= args.max_num_images:
                break
    finally:
        bar.close()

    print("done: %d image(s) in %s" % (written, output_root))


if __name__ == '__main__':
    main()
