import os, sys, argparse, json
import random
import csv
import qrcode
from PIL import Image
import numpy as np
from scipy.stats import bernoulli
from helper_functions import find_files
from extract_leads import get_paper_ecg
from HandwrittenText.generate import get_handwritten
from CreasesWrinkles.creases import get_creased
from ImageAugmentation.augment import get_augment
from PaperCrumple.crumple import get_crumpled, light_azimuth
from CameraOptics.optics import get_blurred
from SceneIllumination.illumination import get_illuminated
from CameraPhotometry.photometry import get_exposed, daylight_gains, mired_to_cct_k
from CameraSensor.sensor import get_resampled, get_sensor_noise, sensor_noise_level
import warnings
from helper_functions import read_config_file

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("-hea", "--header_file", type=str, required=True)
    parser.add_argument("-o", "--output_directory", type=str, required=True)
    parser.add_argument("-se", "--seed", type=int, required=False, default=-1)
    parser.add_argument("-st", "--start_index", type=int, required=True, default=-1)
    parser.add_argument("--num_leads", type=str, default="twelve")
    parser.add_argument("--config_file", type=str, default="config.yaml")

    parser.add_argument("-r", "--resolution", type=int, required=False, default=200)
    parser.add_argument("--pad_inches", type=int, required=False, default=0)
    parser.add_argument("-ph", "--print_header", action="store_true", default=False)
    parser.add_argument("--num_columns", type=int, default=-1)
    parser.add_argument("--full_mode", type=str, default="II")
    parser.add_argument("--mask_unplotted_samples", action="store_true", default=False)
    parser.add_argument("--add_qr_code", action="store_true", default=False)

    parser.add_argument("-l", "--link", type=str, required=False, default="")
    parser.add_argument("-n", "--num_words", type=int, required=False, default=5)
    parser.add_argument("--x_offset", dest="x_offset", type=int, default=30)
    parser.add_argument("--y_offset", dest="y_offset", type=int, default=30)
    parser.add_argument(
        "--hws", dest="handwriting_size_factor", type=float, default=0.2
    )

    parser.add_argument("-ca", "--crease_angle", type=int, default=90)
    parser.add_argument("-nv", "--num_creases_vertically", type=int, default=10)
    parser.add_argument("-nh", "--num_creases_horizontally", type=int, default=10)

    parser.add_argument("-rot", "--rotate", type=int, default=0)
    parser.add_argument("-noise", "--noise", type=int, default=50)
    parser.add_argument("-c", "--crop", type=float, default=0.01)
    parser.add_argument("-t", "--temperature", type=int, default=40000)

    parser.add_argument("--random_resolution", action="store_true", default=False)
    parser.add_argument("--random_padding", action="store_true", default=False)
    parser.add_argument("--random_grid_color", action="store_true", default=False)
    parser.add_argument("--standard_grid_color", type=int, default=5)
    parser.add_argument("--calibration_pulse", type=float, default=1)
    parser.add_argument("--random_grid_present", type=float, default=1)
    parser.add_argument("--random_print_header", type=float, default=0)
    parser.add_argument("--random_bw", type=float, default=0)
    parser.add_argument("--remove_lead_names", action="store_false", default=True)
    parser.add_argument("--lead_name_bbox", action="store_true", default=False)
    parser.add_argument("--store_config", type=int, nargs="?", const=1, default=0)
    parser.add_argument("--store_gridpoints", action="store_true", default=False)

    parser.add_argument("--deterministic_offset", action="store_true", default=False)
    parser.add_argument("--deterministic_num_words", action="store_true", default=False)
    parser.add_argument("--deterministic_hw_size", action="store_true", default=False)

    parser.add_argument("--deterministic_angle", action="store_true", default=False)
    parser.add_argument("--deterministic_vertical", action="store_true", default=False)
    parser.add_argument(
        "--deterministic_horizontal", action="store_true", default=False
    )

    parser.add_argument("--deterministic_rot", action="store_true", default=False)
    parser.add_argument("--deterministic_noise", action="store_true", default=False)
    parser.add_argument("--deterministic_crop", action="store_true", default=False)
    parser.add_argument("--deterministic_temp", action="store_true", default=False)
    parser.add_argument("--deterministic_crumple", action="store_true", default=False)
    parser.add_argument("--deterministic_blur", action="store_true", default=False)
    parser.add_argument("--deterministic_illum", action="store_true", default=False)
    parser.add_argument("--deterministic_vignette", action="store_true", default=False)
    parser.add_argument("--deterministic_black_point", action="store_true", default=False)
    parser.add_argument("--deterministic_white_point", action="store_true", default=False)

    parser.add_argument("--trace_thickness_mm", type=float, default=None)
    parser.add_argument("--trace_thickness_jitter", type=float, default=0.15)
    parser.add_argument("--trace_dropout_rate", type=float, default=0.0)
    parser.add_argument("--trace_dropout_length_mm", type=float, default=0.5)

    # Vertical gap between a grid lead's name (e.g. "V1") and its trace baseline.
    # Default None reproduces the upstream fixed 7 mm; jitter is a half width in mm,
    # drawn once per frame and shared by every grid label in that frame.
    parser.add_argument("--lead_name_gap_mm", type=float, default=None)
    parser.add_argument("--lead_name_gap_jitter_mm", type=float, default=0.0)

    # Blank space between columns of a grid layout, replacing the upstream black
    # lead-separator tick. Default None keeps the columns flush and the tick drawn;
    # --column_gap_jitter_mm makes each seam's gap an independent per-frame draw.
    parser.add_argument("--column_gap_mm", type=float, default=None)
    parser.add_argument("--column_gap_jitter_mm", type=float, default=0.0)

    parser.add_argument("--crumple_amplitude", type=float, default=0.0)
    parser.add_argument("--crumple_scale_cm", type=float, default=8.0)
    # Shared with the illumination stage below: one light direction per image.
    parser.add_argument("--illum_azimuth_deg", type=float, default=-1)

    parser.add_argument("--blur_sigma", type=float, default=0.0)

    parser.add_argument("--illum_strength", type=float, default=0.0)

    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--exposure_jitter_stops", type=float, default=0.0)

    parser.add_argument("--wb_r", type=float, default=1.0)
    parser.add_argument("--wb_b", type=float, default=1.0)
    parser.add_argument("--wb_mired_jitter", type=float, default=0.0)

    parser.add_argument("--vignette", type=float, default=0.0)

    parser.add_argument("--contrast", type=float, default=1.0)
    parser.add_argument("--contrast_jitter_log2", type=float, default=0.0)

    parser.add_argument("--black_point", type=float, default=0.0)
    parser.add_argument("--white_point", type=float, default=1.0)

    parser.add_argument("--saturation", type=float, default=1.0)
    parser.add_argument("--saturation_jitter_log2", type=float, default=0.0)

    parser.add_argument("--hue_rotation", type=float, default=0.0)
    parser.add_argument("--hue_rotation_jitter_deg", type=float, default=0.0)

    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--output_width", type=int, default=0)
    parser.add_argument("--output_height", type=int, default=0)

    parser.add_argument("--sensor_noise", type=float, default=0.0)
    parser.add_argument("--sensor_noise_jitter_log2", type=float, default=0.0)

    parser.add_argument("--fully_random", action="store_true", default=False)
    parser.add_argument("--hw_text", action="store_true", default=False)
    parser.add_argument("--wrinkles", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--lead_bbox", action="store_true", default=False)

    return parser


def writeCSV(args):
    csv_file_path = os.path.join(args.output_directory, "Coordinates.csv")
    if os.path.isfile(csv_file_path) == False:
        with open(csv_file_path, "a") as ground_truth_file:
            writer = csv.writer(ground_truth_file)
            if args.start_index != -1:
                writer.writerow(
                    ["Filename", "class", "x_center", "y_center", "width", "height"]
                )

    grid_file_path = os.path.join(args.output_directory, "gridsizes.csv")
    if os.path.isfile(grid_file_path) == False:
        with open(grid_file_path, "a") as gridsize_file:
            writer = csv.writer(gridsize_file)
            if args.start_index != -1:
                writer.writerow(
                    ["filename", "xgrid", "ygrid", "lead_name", "start", "end"]
                )


def run_single_file(args):
    if args.store_gridpoints and not args.store_config:
        # The sidecar .npy files and the JSON mirror both need rec_tail, which is only
        # computed below when --store_config is truthy (same reason --augment already
        # requires it - see CLAUDE.md's "Known traps"). Fail loudly here instead of
        # letting store_gridpoints silently produce nothing, or crash later on an
        # undefined rec_tail.
        raise SystemExit(
            "--store_gridpoints requires --store_config (1 or 2): the gridpoint "
            "annotation is written into the per-frame JSON and its .npy sidecars share "
            "that JSON's base filename."
        )

    if hasattr(args, "st") == True:
        random.seed(args.seed)
        args.encoding = args.input_file

    filename = args.input_file
    header = args.header_file
    resolution = (
        random.choice(range(50, args.resolution + 1))
        if (args.random_resolution)
        else args.resolution
    )
    padding = (
        random.choice(range(0, args.pad_inches + 1))
        if (args.random_padding)
        else args.pad_inches
    )

    # Supersampling: the whole chain up to the sensor is rendered at a multiple of the
    # delivered resolution and integrated back down in get_resampled, immediately before
    # the sensor noise. This page is the case that needs it - a 0.30 mm trace is 1.8 px at
    # 300 dpi and 0.9 px at 150, and the 1 mm minor grid is 1.2 to 2.4 px, so rendering
    # straight at the delivered resolution puts every line at or under the sampling limit.
    # --resolution KEEPS MEANING THE OUTPUT dpi. Only the render moves, so the 150-300
    # range in the batch configs still describes the images that come out and every earlier
    # evaluation under out/ stays comparable.
    # Everything the chain expresses in MILLIMETRES of paper - trace_thickness_mm,
    # trace_dropout_length_mm, crumple_scale_cm - converts through whatever dpi it is
    # handed and therefore follows this for free. blur_sigma is the one exception and is
    # scaled explicitly below.
    supersample = max(int(args.supersample), 1)
    render_resolution = resolution * supersample

    papersize = ""
    lead = args.remove_lead_names

    bernoulli_dc = bernoulli(args.calibration_pulse)
    bernoulli_bw = bernoulli(args.random_bw)
    bernoulli_grid = bernoulli(args.random_grid_present)
    if args.print_header:
        bernoulli_add_print = bernoulli(1)
    else:
        bernoulli_add_print = bernoulli(args.random_print_header)

    font = os.path.join("Fonts", random.choice(os.listdir("Fonts")))

    if args.random_bw == 0:
        if args.random_grid_color == False:
            standard_colours = args.standard_grid_color
        else:
            standard_colours = -1
    else:
        standard_colours = False

    configs = read_config_file(os.path.join(os.getcwd(), args.config_file))

    out_array = get_paper_ecg(
        input_file=filename,
        header_file=header,
        configs=configs,
        mask_unplotted_samples=args.mask_unplotted_samples,
        start_index=args.start_index,
        store_configs=args.store_config,
        store_text_bbox=args.lead_name_bbox,
        output_directory=args.output_directory,
        resolution=render_resolution,
        papersize=papersize,
        add_lead_names=lead,
        add_dc_pulse=bernoulli_dc,
        add_bw=bernoulli_bw,
        show_grid=bernoulli_grid,
        add_print=bernoulli_add_print,
        pad_inches=padding,
        font_type=font,
        standard_colours=standard_colours,
        full_mode=args.full_mode,
        bbox=args.lead_bbox,
        columns=args.num_columns,
        seed=args.seed,
        trace_thickness_mm=args.trace_thickness_mm,
        trace_thickness_jitter=args.trace_thickness_jitter,
        trace_dropout_rate=args.trace_dropout_rate,
        trace_dropout_length_mm=args.trace_dropout_length_mm,
        lead_name_gap_mm=args.lead_name_gap_mm,
        lead_name_gap_jitter_mm=args.lead_name_gap_jitter_mm,
        column_gap_mm=args.column_gap_mm,
        column_gap_jitter_mm=args.column_gap_jitter_mm,
        store_gridpoints=args.store_gridpoints,
    )

    for out in out_array:
        if args.store_config:
            rec_tail, extn = os.path.splitext(out)
            with open(rec_tail + ".json", "r") as file:
                json_dict = json.load(file)
        else:
            json_dict = None
        if args.fully_random:
            hw_text = random.choice((True, False))
            wrinkles = random.choice((True, False))
            augment = random.choice((True, False))
        else:
            hw_text = args.hw_text
            wrinkles = args.wrinkles
            augment = args.augment

        # Handwritten text addition
        if hw_text:
            num_words = (
                args.num_words
                if (args.deterministic_num_words)
                else random.choice(range(2, args.num_words + 1))
            )
            x_offset = (
                args.x_offset
                if (args.deterministic_offset)
                else random.choice(range(1, args.x_offset + 1))
            )
            y_offset = (
                args.y_offset
                if (args.deterministic_offset)
                else random.choice(range(1, args.y_offset + 1))
            )

            out = get_handwritten(
                link=args.link,
                num_words=num_words,
                input_file=out,
                output_dir=args.output_directory,
                x_offset=x_offset,
                y_offset=y_offset,
                handwriting_size_factor=args.handwriting_size_factor,
                bbox=args.lead_bbox,
            )
        else:
            num_words = 0
            x_offset = 0
            y_offset = 0

        if args.store_config == 2:
            json_dict["handwritten_text"] = bool(hw_text)
            json_dict["num_words"] = num_words
            json_dict["x_offset_for_handwritten_text"] = x_offset
            json_dict["y_offset_for_handwritten_text"] = y_offset

        if wrinkles:
            ifWrinkles = True
            ifCreases = True
            crease_angle = (
                args.crease_angle
                if (args.deterministic_angle)
                else random.choice(range(0, args.crease_angle + 1))
            )
            num_creases_vertically = (
                args.num_creases_vertically
                if (args.deterministic_vertical)
                else random.choice(range(1, args.num_creases_vertically + 1))
            )
            num_creases_horizontally = (
                args.num_creases_horizontally
                if (args.deterministic_horizontal)
                else random.choice(range(1, args.num_creases_horizontally + 1))
            )
            out = get_creased(
                out,
                output_directory=args.output_directory,
                ifWrinkles=ifWrinkles,
                ifCreases=ifCreases,
                crease_angle=crease_angle,
                num_creases_vertically=num_creases_vertically,
                num_creases_horizontally=num_creases_horizontally,
                bbox=args.lead_bbox,
            )
        else:
            crease_angle = 0
            num_creases_horizontally = 0
            num_creases_vertically = 0

        if args.store_config == 2:
            json_dict["wrinkles"] = bool(wrinkles)
            json_dict["crease_angle"] = crease_angle
            json_dict["number_of_creases_horizontally"] = num_creases_horizontally
            json_dict["number_of_creases_vertically"] = num_creases_vertically

        # Paper crumpling: the last substrate stage, before the camera stages. A single
        # height field drives both the deformation and its shading, so creases and
        # their shadows cannot drift apart.
        # The amplitude on the command line is a maximum, sampled per image unless
        # --deterministic_crumple, following the idiom of -ca and -rot. It is only
        # drawn when the parameter is active, so that the default of 0 leaves the
        # global random sequence, and therefore the render, untouched.
        if args.crumple_amplitude > 0 and args.deterministic_crumple == False:
            crumple_amplitude = random.uniform(0, args.crumple_amplitude)
        else:
            crumple_amplitude = args.crumple_amplitude
        illum_azimuth_deg = light_azimuth(
            args.illum_azimuth_deg, out, seed=args.seed, start_index=args.start_index
        )

        if crumple_amplitude > 0:
            out = get_crumpled(
                out,
                resolution=render_resolution,
                crumple_amplitude=crumple_amplitude,
                crumple_scale_cm=args.crumple_scale_cm,
                illum_azimuth_deg=illum_azimuth_deg,
                seed=args.seed,
                start_index=args.start_index,
                json_dict=json_dict,
            )

        if args.store_config == 2:
            json_dict["crumple_amplitude"] = round(crumple_amplitude, 4)
            json_dict["crumple_scale_cm"] = args.crumple_scale_cm
            json_dict["illum_azimuth_deg"] = round(illum_azimuth_deg, 2)

        # Optical blur: the first camera stage, acting on the scene as already formed.
        # It has to run before get_augment, whose gaussian noise belongs to the sensor
        # and therefore lands on top of the blur; blurring after the noise would smooth
        # the noise away.
        # The sigma on the command line is a maximum, sampled per image unless
        # --deterministic_blur, following the idiom of -ca, -rot and --crumple_amplitude.
        # It is only drawn when the parameter is active, so that the default of 0 leaves
        # the global random sequence, and therefore the render, untouched.
        if args.blur_sigma > 0 and args.deterministic_blur == False:
            blur_sigma = random.uniform(0, args.blur_sigma)
        else:
            blur_sigma = args.blur_sigma

        if blur_sigma > 0:
            # Multiplied by the supersample factor for the render, and ONLY for the
            # render. blur_sigma is the single parameter of this chain denominated in
            # pixels rather than in millimetres of paper - optics.py says so explicitly,
            # because a defocus is a property of the camera and is measured on its sensor
            # - so it is the single parameter that does not follow the render resolution
            # on its own. Left unscaled, the same flag would be a THIRD of the blur at
            # supersample 3 once the page is integrated back down, and every figure the
            # earlier evaluations calibrated against would silently change meaning.
            # The flag therefore keeps meaning px OF THE DELIVERED IMAGE, which is also
            # what makes blur_sigma_mm below invariant: the sigma and the resolution it is
            # divided by are both the output ones.
            out = get_blurred(out, blur_sigma=blur_sigma * supersample)

        if args.store_config == 2:
            # blur_sigma is in px at the OUTPUT resolution, which is what the parameter
            # means and what calibration bisects on. The mm equivalent is recorded
            # beside it so that annotations from renders made at different dpi can still
            # be compared in paper space.
            json_dict["blur_sigma"] = round(blur_sigma, 4)
            json_dict["blur_sigma_mm"] = round(blur_sigma * 25.4 / resolution, 4)

        # Non-uniform illumination: the first photometric stage. Window light or a lamp
        # off to one side, multiplied in linear light. It acts on the scene as the camera
        # already saw it - deformed and defocused - and before the sensor stages of
        # get_augment, whose gaussian noise and colour temperature belong to the sensor
        # and not to the room.
        # The azimuth is the one already resolved for the crumple above, not a new draw:
        # the shadow on the far side of a fold and the dim end of the page have to agree
        # on where the light is.
        # The strength on the command line is a maximum, sampled per image unless
        # --deterministic_illum, following the idiom of -ca, -rot, --crumple_amplitude and
        # --blur_sigma. It is only drawn when the parameter is active, so that the default
        # of 0 leaves the global random sequence, and therefore the render, untouched.
        if args.illum_strength > 0 and args.deterministic_illum == False:
            illum_strength = random.uniform(0, args.illum_strength)
        else:
            illum_strength = args.illum_strength

        if illum_strength > 0:
            out = get_illuminated(
                out,
                illum_strength=illum_strength,
                illum_azimuth_deg=illum_azimuth_deg,
            )

        if args.store_config == 2:
            # The azimuth is already recorded by the crumple block above, which resolves
            # it for both stages.
            json_dict["illum_strength"] = round(illum_strength, 4)

        # Exposure: the gain the camera applied to the light the illumination above put
        # on the sheet, multiplied in linear light. It is the only stage in the chain
        # meant to move the page brightness - the crumple and the illumination both undo
        # their own effect on the mean, and the blur conserves energy - so that lum_mean
        # has a single owner.
        # It does NOT follow the "the flag is a maximum, drawn from uniform(0, value)"
        # idiom of -ca, -rot, --crumple_amplitude, --blur_sigma and --illum_strength,
        # because a gain is neutral at 1.0 and not at 0, and its range is two sided.
        # --exposure is the gain itself, which is what a bisection calibration moves;
        # --exposure_jitter_stops is the half width of a per image draw around it, in
        # STOPS rather than in gain, because exposure error in a real photograph is
        # symmetric in stops and not in the multiplier. A jitter of 0 is therefore
        # already the deterministic mode, and no --deterministic_exposure is added for
        # it. The draw only happens when the jitter is active, so that the default leaves
        # the global random sequence - and therefore the font, the grid colour and every
        # augment draw below - untouched.
        if args.exposure_jitter_stops > 0:
            exposure = args.exposure * 2.0 ** random.uniform(
                -args.exposure_jitter_stops, args.exposure_jitter_stops
            )
        else:
            exposure = args.exposure

        # White balance: the colour of the light, as two per-channel gains in linear
        # light. It is FUSED into the exposure stage rather than given one of its own,
        # because every stage boundary here is a uint8 PNG and a separate stage would
        # spend half a level of quantisation on a cast that is only about ten levels
        # wide. Physically the two are one diagonal gain matrix anyway, which is what the
        # roteiro means by "applied alongside exposure".
        # --wb_r and --wb_b are the gains themselves, and they are what a bisection
        # calibration moves. --wb_mired_jitter is the half width of a per image draw
        # around them, in MIREDS - reciprocal megakelvin - rather than in kelvin, for the
        # same reason the exposure jitter is in stops: 500 K is an enormous shift at
        # 3000 K and invisible at 15000 K, while a mired is roughly the same perceived
        # step everywhere.
        # The draw moves the pair ALONG THE DAYLIGHT LOCUS, which is the point of it. The
        # roteiro requires the two gains be sampled in a correlated way and not
        # independently, because a real illuminant has one degree of freedom: warm light
        # is high r AND low b, and the (high r, high b) corner two independent uniforms
        # would produce is a lamp that does not exist. This is also why the parameter is
        # NOT randomised through the runner's randomize: block, which draws every key on
        # its own - the inversion from --exposure is deliberate and run_batch_from_config
        # refuses the combination rather than silently decorrelating the pair.
        # A jitter of 0 is already the deterministic mode, so no --deterministic_wb is
        # added, exactly as none was added for the exposure. The draw only happens when
        # the jitter is active, so that the default leaves the global random sequence -
        # and therefore every augment draw below - untouched.
        if args.wb_mired_jitter > 0:
            wb_mired_offset = random.uniform(
                -args.wb_mired_jitter, args.wb_mired_jitter
            )
            jitter_r, jitter_b = daylight_gains(wb_mired_offset)
        else:
            wb_mired_offset = 0.0
            jitter_r, jitter_b = 1.0, 1.0
        wb_r = args.wb_r * jitter_r
        wb_b = args.wb_b * jitter_b

        # Vignette: the radial luminance falloff of the optical system, fused into the
        # exposure stage. Unlike the illumination gradient it is SYMMETRIC - it is a
        # property of the lens rather than of where the light is - which is why the two
        # are separate parameters and not one, and why nothing here reads the azimuth.
        # It follows the idiom of -ca, -rot, --crumple_amplitude, --blur_sigma and
        # --illum_strength rather than the jitter idiom of --exposure and the white
        # balance: the value on the command line is a MAXIMUM, sampled per image unless
        # --deterministic_vignette. Which idiom applies is decided by the neutral value,
        # not by taste - a jitter exists for exposure and white balance because their
        # neutral is 1.0 and their range runs to both sides of it, while a falloff is
        # neutral at 0 and one sided, so uniform(0, max) covers it.
        # Drawn only when the parameter is active, so the default of 0 leaves the global
        # random sequence - and therefore every augment draw below - untouched.
        if args.vignette > 0 and args.deterministic_vignette == False:
            vignette = random.uniform(0, args.vignette)
        else:
            vignette = args.vignette

        # Contrast: the tone curve, and the first stage of the chain that acts in
        # DISPLAY space rather than in linear light - which is where the roteiro puts
        # it, and where contrast_rms is measured. It is fused into get_exposed with the
        # three above so that the curve meets the [0,1] clip of the uint8 boundary once
        # instead of twice.
        # It follows the jitter idiom of --exposure and the white balance and not the
        # uniform(0, max) idiom of --vignette, by the rule this codebase already states:
        # which idiom applies is decided by the NEUTRAL VALUE, not by taste. Contrast is
        # neutral at 1.0 and its range runs to both sides of it, so uniform(0, max) does
        # not cover it and there is no --deterministic_contrast - a jitter of 0 is
        # already the deterministic mode, exactly as for the exposure.
        # The draw is in LOG2 of the contrast rather than in the multiplier itself, for
        # the same reason the exposure jitter is in stops and the white balance jitter in
        # mireds: the roteiro's own 0.6-1.8 range is nearly symmetric in log2
        # (-0.74 / +0.85) and badly asymmetric in the multiplier, so a symmetric draw on
        # the multiplier would spend most of its width on one side of neutral.
        # Drawn only when the jitter is active, so the default leaves the global random
        # sequence - and therefore every augment draw below - untouched.
        if args.contrast_jitter_log2 > 0:
            contrast = args.contrast * 2.0 ** random.uniform(
                -args.contrast_jitter_log2, args.contrast_jitter_log2
            )
        else:
            contrast = args.contrast

        # Shadow and highlight clipping: the black and white point, the last stage of the
        # photometric chain and the last thing get_exposed does. out = clip((in - bp) /
        # (wp - bp), 0, 1), in display space, after the tone curve.
        # It is the FIRST STAGE IN THE CHAIN THAT DOES NOT PUT lum_mean BACK, and the
        # reason is algebraic rather than an oversight: the levels map is a gain of
        # 1/(wp - bp) plus an offset, and the contrast curve is a gain of c plus an offset
        # already solved for a constant mean, so solving this one the same way would
        # collapse the pair onto contrast = 1/(wp - bp) exactly. What this parameter owns
        # is the CLIP - clip_shadows_pct and clip_highlights_pct - and not the mean. The
        # proof is written out in full in CameraPhotometry/photometry.py.
        #
        # BOTH follow the uniform idiom of --vignette rather than the jitter idiom of
        # --exposure and --contrast, by the rule this codebase already states: which idiom
        # applies is decided by the NEUTRAL VALUE, not by taste. Both are one sided.
        # --black_point is neutral at 0 and grows, so the value is a maximum and the draw
        # is uniform(0, max), exactly as the vignette's is.
        # --white_point is neutral at 1.0 and SHRINKS, so it is the same idiom mirrored:
        # the value is a MINIMUM and the draw is uniform(min, 1.0). Drawing uniform(0, wp)
        # here would put most of the batch near black, which is not what a highlight clip
        # does at any setting.
        # Drawn only when the parameter is active, so the defaults leave the global random
        # sequence - and therefore every augment draw below - untouched.
        if args.black_point > 0 and args.deterministic_black_point == False:
            black_point = random.uniform(0, args.black_point)
        else:
            black_point = args.black_point

        if args.white_point < 1 and args.deterministic_white_point == False:
            white_point = random.uniform(args.white_point, 1.0)
        else:
            white_point = args.white_point

        # Chroma scaling: the saturation, and the last thing get_exposed does - after the
        # clipping points, immediately before the uint8 cast, which is the roteiro's own
        # order for the chain (tone curve, then clipping, then colour).
        # Scaled in CIELAB and not in HSV, which the roteiro requires by name: a and b are
        # zero on the neutral axis, so the paper white the exposure and the white balance
        # just settled stays exactly where it was at every value of this parameter, and only
        # the pixels that already carry colour move. On an ECG sheet that is the grid.
        # It is also the first stage in this chain since the crumple that needs NO mean
        # preserving solve, and for a reason rather than by omission: scaling a and b leaves
        # L untouched by construction, so exposure keeps sole ownership of lum_mean without
        # anything being restored. The full argument is in CameraPhotometry/photometry.py.
        #
        # It follows the jitter idiom of --exposure and --contrast and not the uniform(0,
        # max) idiom of --vignette and the two clipping points, by the rule this codebase
        # already states: which idiom applies is decided by the NEUTRAL VALUE, not by taste.
        # Saturation is neutral at 1.0 and its range runs to both sides of it, so
        # uniform(0, max) does not cover it and there is no --deterministic_saturation - a
        # jitter of 0 is already the deterministic mode.
        # The draw is in LOG2 of the scale rather than in the multiplier itself, for the same
        # reason the exposure jitter is in stops and the contrast jitter in log2: chroma
        # scaling is multiplicative, so half and double are the same size of step and a
        # symmetric draw on the multiplier would not be symmetric on the thing being scaled.
        # Drawn only when the jitter is active, so the default leaves the global random
        # sequence - and therefore every augment draw below - untouched.
        if args.saturation_jitter_log2 > 0:
            saturation = args.saturation * 2.0 ** random.uniform(
                -args.saturation_jitter_log2, args.saturation_jitter_log2
            )
        else:
            saturation = args.saturation

        # Hue rotation: the angle a* and b* are turned through, sharing the CIELAB block
        # and the single Lab round trip with the saturation above. The two are linear maps
        # of the same plane and they COMMUTE - s*I*R = R*s*I - so there is no order between
        # them to decide and no second conversion to pay for.
        # What it models is narrower than the name: the white balance moves the illuminant
        # along the daylight locus, which is one degree of freedom, and the real
        # photographs sit beside that curve rather than on it. This is the residual, not a
        # colour control of its own.
        #
        # It follows the jitter idiom of --exposure, --contrast and --saturation rather
        # than the uniform(0, max) idiom of --vignette and the two clipping points, but NOT
        # for the reason those three do. The rule this codebase states is that the neutral
        # value decides, and here the neutral is 0 like the vignette's - what rules out
        # uniform(0, max) is that the range is BILATERAL: a rotation runs to both sides of
        # neutral and a one-sided draw would only ever turn the grid one way. So there is
        # no --deterministic_hue_rotation either; a jitter of 0 is already the
        # deterministic mode, exactly as for the three above.
        # The draw is ADDITIVE IN DEGREES and not in log2, which is where this parameter
        # parts company with the exposure, contrast and saturation jitters. Those three are
        # multiplicative, so half and double are the same size of step and only the log is
        # symmetric. Rotations compose by ADDITION - two turns of 4 degrees are one of 8,
        # not one of 16 - so degrees are already the axis a symmetric draw belongs on.
        # Drawn only when the jitter is active, so the default leaves the global random
        # sequence - and therefore every augment draw below - untouched.
        if args.hue_rotation_jitter_deg > 0:
            hue_rotation = args.hue_rotation + random.uniform(
                -args.hue_rotation_jitter_deg, args.hue_rotation_jitter_deg
            )
        else:
            hue_rotation = args.hue_rotation

        out = get_exposed(out, exposure=exposure, wb_r=wb_r, wb_b=wb_b,
                          vignette=vignette, contrast=contrast,
                          black_point=black_point, white_point=white_point,
                          saturation=saturation, hue_rotation=hue_rotation)

        if args.store_config == 2:
            # Recorded in stops beside the gain: a batch varies exposure log uniformly,
            # so the stops are the axis its distribution is actually flat on, and the
            # figure that compares against a photographic exposure error.
            json_dict["exposure"] = round(exposure, 4)
            json_dict["exposure_stops"] = round(float(np.log2(exposure)), 4)
            json_dict["wb_r"] = round(wb_r, 4)
            json_dict["wb_b"] = round(wb_b, 4)
            # The mired offset and its kelvin equivalent describe the DRAW, which is on
            # the locus by construction, and they are recorded only when there was one.
            # A pair set by hand is a free point in the (r, b) plane and generally sits
            # off the locus, where there is no single colour temperature to report:
            # projecting a 2D point onto a 1D curve would fill the annotations with a
            # plausible-looking number that is not true of the image.
            if args.wb_mired_jitter > 0:
                json_dict["wb_mired_offset"] = round(wb_mired_offset, 3)
                json_dict["wb_cct_k"] = round(mired_to_cct_k(wb_mired_offset), 1)
            # The strength actually applied, not the maximum on the command line.
            # No second figure beside it: unlike the exposure there is no other axis
            # this one is naturally flat on, and the geometry it implies is closed form
            # anyway - the mask is 1/(1 - v/3) at the centre and (1 - v)/(1 - v/3) at
            # the corners, so a reader with this number has the whole falloff.
            json_dict["vignette"] = round(vignette, 4)
            # The value applied, following the vignette. The offset the stage solves to
            # hold lum_mean across the curve is not recorded beside it: it is a function
            # of this number and the page mean, and it is the evaluation sweep - not the
            # per-image annotation - that has a use for it.
            json_dict["contrast"] = round(contrast, 4)
            # The two values applied, following the contrast. No derived figure beside
            # them: unlike the exposure there is no other axis these are naturally flat
            # on, and the map they imply is closed form anyway - a reader with the pair
            # has the whole curve, including the display gain 1/(wp - bp).
            json_dict["black_point"] = round(black_point, 4)
            json_dict["white_point"] = round(white_point, 4)
            # The scale applied, following the clipping points and closing stage D. No
            # derived figure beside it: unlike the exposure there is no other axis this one
            # is naturally flat on, and what it implies is closed form anyway - every a* and
            # b* in the image is this multiple of what the render produced, and the neutral
            # axis is fixed, so a reader with this number has the whole transform.
            json_dict["saturation"] = round(saturation, 4)
            # The angle applied, in degrees, closing stage D. No derived figure beside it,
            # for the reason the saturation gives: every a* and b* in the image is this
            # rotation of what the render produced and the neutral axis is fixed, so a
            # reader with this number has the whole transform. Recorded in degrees rather
            # than radians because degrees are the unit the flag, the range and the
            # roteiro all use.
            json_dict["hue_rotation"] = round(hue_rotation, 4)

        # The sampling grid of the sensor, closing stage E-13. The page has been rendered,
        # deformed, defocused and graded at render_resolution; here it is integrated down
        # with INTER_AREA onto the pixels that are actually delivered.
        # POSITION IS THE WHOLE POINT and it is fixed at both ends. After every stage that
        # belongs to the scene and to the lens, because those act on a continuous optical
        # image and the sensor is what discretises it. BEFORE get_augment, because the
        # gaussian noise in there originates ON this grid: noise added at render
        # resolution and then downscaled by 3 comes out with a third of the amplitude and
        # a correlation length it should not have, and sensor_noise, the next increment,
        # would be calibrating against an artefact of the supersample factor.
        # THAT LAST CLAUSE READ THE SLOT WRONG, and E-14 corrected it. The requirement it
        # states is real - the noise has to originate on the delivered grid - but this is
        # not the only point that satisfies it, and it is not the right one: get_augment
        # RESAMPLES, through iaa.Affine(rotate) and iaa.Crop, so noise injected here would
        # be smoothed by an angle drawn per image. get_sensor_noise therefore runs at the
        # very end of this function, still on the delivered grid and no longer upstream of
        # a resample. Everything above about the DOWNSCALE belonging before get_augment
        # stands unchanged.
        # Before get_augment for a second, mechanical reason: it reads h, w from the image
        # and uses [h/2, w/2] as the rotation origin for the annotations, so the pixels and
        # the stored geometry have to already agree on the frame.
        # At supersample 1 with no explicit output size this is a no op that does not even
        # reopen the file - see get_resampled.
        out = get_resampled(out, supersample=supersample,
                            output_width=args.output_width,
                            output_height=args.output_height,
                            json_dict=json_dict)

        if args.store_config == 2:
            # The factor the page was oversampled by, and the size it was delivered at.
            # width and height are already recorded by the render and rewritten by the
            # stage above, so what is added here is the provenance the delivered figures
            # no longer carry: at what dpi the page was actually drawn before it was
            # integrated down. Two images of identical size and very different edge
            # statistics differ by exactly this number.
            json_dict["supersample"] = supersample
            json_dict["render_resolution"] = render_resolution

        if augment:
            noise = (
                args.noise
                if (args.deterministic_noise)
                else random.choice(range(1, args.noise + 1))
            )

            if not args.lead_bbox:
                do_crop = random.choice((True, False))
                if do_crop:
                    crop = args.crop
                else:
                    crop = args.crop
            else:
                crop = 0
            # --temperature and --deterministic_temp were declared by upstream and never
            # read, which put an UNCONTROLLED colour temperature at the end of the chain:
            # measured, iaa.ChangeColorTemperature turns white paper into [255,137,18] at
            # 2000 K and [168,197,255] at 20000 K, and the draw below picks one of those
            # two extremes with even odds and nothing in between. Across the ten images of
            # the previous batch that produced a bimodal spread of 75 units in lab_b, while
            # wb_r and wb_b over their whole documented range move it by about 5.
            # Two stages cannot own the colour of one photograph, for the same reason
            # lum_mean has a single owner and the crumple and the illumination share one
            # light azimuth. Reading the flag hands that ownership to the white balance
            # above, where it is a physical gain in linear light on the daylight locus
            # rather than a coin toss between orange and blue.
            # The default is False, so the draw below is untouched and the whole chain
            # still reproduces bit for bit with the flag left off. Note that switching it
            # ON consumes two fewer values from the global random sequence, so the
            # rotation, crop and noise of get_augment shift with it - which is expected,
            # and the reason a batch run with deterministic_temp is not comparable image
            # by image with one run without it.
            if args.deterministic_temp:
                temp = args.temperature
            else:
                blue_temp = random.choice((True, False))

                if blue_temp:
                    temp = random.choice(range(2000, 4000))
                else:
                    temp = random.choice(range(10000, 20000))
            rotate = args.rotate
            out = get_augment(
                out,
                output_directory=args.output_directory,
                rotate=args.rotate,
                noise=noise,
                crop=crop,
                temperature=temp,
                bbox=args.lead_bbox,
                store_text_bounding_box=args.lead_name_bbox,
                json_dict=json_dict,
            )

        else:
            crop = 0
            temp = 0
            rotate = 0
            noise = 0
        # The amplitude this sheet is grained with, resolved before the JSON is written
        # even though the stage itself runs after the QR block below. Nothing between the
        # two touches it, and the annotation has to carry the value that was applied.
        sensor_noise = sensor_noise_level(
            args.sensor_noise,
            args.sensor_noise_jitter_log2,
            out,
            seed=args.seed,
            start_index=args.start_index,
        )

        if args.store_config == 2:
            json_dict["augment"] = bool(augment)
            json_dict["crop"] = crop
            json_dict["temperature"] = temp
            json_dict["rotate"] = rotate
            # The legacy imgaug noise, kept in the annotation because it is still a flag
            # and a reader has to be able to tell which of the two grained the image.
            # Set it to 0 with deterministic_noise on and sensor_noise owns the page.
            json_dict["noise"] = noise
            # The amplitude applied, in levels of 0-255, closing stage E-14. The jitter is
            # not recorded beside it: this IS the drawn value, and the centre it came from
            # says nothing about this image that this number does not say better.
            json_dict["sensor_noise"] = round(sensor_noise, 4)
            if augment and crop > 0:
                # get_augment's crop step (iaa.Crop(percent=...,keep_size=True)) is a
                # crop AND a resize back to canvas size, i.e. an offset AND a scale on
                # every coordinate. Previously only the rotation was corrected and
                # plotted_pixels silently drifted out of alignment with the image
                # whenever crop > 0; get_augment now applies the same crop correction
                # gridpoints gets. Recorded here because it changes what earlier
                # annotations from this generator mean.
                json_dict["plotted_pixels_crop_corrected"] = True

        if args.store_gridpoints:
            # gridpoints_mask is computed once, here, against the FINAL frame - after
            # crumple, the E-13 resample and get_augment have all had their say - rather
            # than threaded through every intermediate stage. A node is visible if its
            # final coordinate falls inside the delivered canvas; nodes that warped or
            # cropped out still carry a real coordinate (never a (0,0) sentinel), just
            # masked False, so rectify_image's dense F.interpolate never gets pulled
            # toward the origin by a hole in the lattice.
            n_rows, n_cols = json_dict["gridpoints_shape"]
            gridpoints_arr = np.asarray(json_dict["gridpoints"], dtype=np.float32).reshape(
                n_rows, n_cols, 2
            )
            cols = gridpoints_arr[:, :, 0]
            rows = gridpoints_arr[:, :, 1]
            mask = (
                (rows >= 0) & (rows < json_dict["height"]) &
                (cols >= 0) & (cols < json_dict["width"])
            )
            json_dict["gridpoints_mask"] = mask.tolist()

            # hengck23's own on-disk convention: (n_rows,n_cols,2) float32 [x,y], at
            # OUTPUT resolution - now the SAME axis order as the JSON mirror above,
            # which also stores [x,y] to match plotted_pixels, so this is a reshape.
            gridpoint_xy = gridpoints_arr
            np.save(rec_tail + ".gridpoint_xy.npy", gridpoint_xy)
            np.save(rec_tail + ".gridpoint_mask.npy", mask)

        if args.store_config:
            # Every coordinate pair this generator writes - plotted_pixels, the four
            # corners of either bounding box, gridpoints - is [x, y]. Recorded here
            # for the same reason plotted_pixels_crop_corrected is: it changes what
            # earlier annotations from this generator mean, and there is no other
            # version marker on the JSON schema.
            json_dict["coordinate_order"] = "xy"
            json_object = json.dumps(json_dict, indent=4)

            with open(rec_tail + ".json", "w") as f:
                f.write(json_object)

        if args.add_qr_code:
            img = np.array(Image.open(out))
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=5,
                border=4,
            )
            qr.add_data(args.encoding)
            qr.make(fit=True)

            qr_img = np.array(qr.make_image(fill_color="black", back_color="white"))
            qr_img_color = np.zeros((qr_img.shape[0], qr_img.shape[1], 3))
            qr_img_color[:, :, 0] = qr_img * 255.0
            qr_img_color[:, :, 1] = qr_img * 255.0
            qr_img_color[:, :, 2] = qr_img * 255.0

            img[: qr_img.shape[0], -qr_img.shape[1] :, :3] = qr_img_color
            img = Image.fromarray(img)
            img.save(out)

        # The noise of the sensor, closing stage E-14 and the chain. LAST, with no
        # exceptions above it: everything before this describes the page, the light or the
        # lens, and this is the instrument reading them. Two of those stages resample
        # after the sampling grid is fixed - iaa.Affine(rotate) and iaa.Crop inside
        # get_augment - so anywhere earlier the grain would be smoothed by a draw of the
        # augment's rather than by this parameter, and the measured noise_sigma would stop
        # being a function of the flag. After the QR stamp for the same reason read the
        # other way: a code pasted onto a grained page and left clean is the one region of
        # the image that would advertise the compositing.
        # The jpeg_q of increment 15 goes after this and nothing else does.
        # At 0 - the default - this does not even open the file. See get_sensor_noise.
        out = get_sensor_noise(
            out,
            sensor_noise=sensor_noise,
            seed=args.seed,
            start_index=args.start_index,
        )

    return len(out_array)


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), sys.argv[0])
    parentPath = os.path.dirname(path)
    os.chdir(parentPath)
    run_single_file(get_parser().parse_args(sys.argv[1:]))
