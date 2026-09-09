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

    parser.add_argument("--trace_thickness_mm", type=float, default=None)
    parser.add_argument("--trace_thickness_jitter", type=float, default=0.15)
    parser.add_argument("--trace_dropout_rate", type=float, default=0.0)
    parser.add_argument("--trace_dropout_length_mm", type=float, default=0.5)

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
        resolution=resolution,
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
                resolution=resolution,
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
            out = get_blurred(out, blur_sigma=blur_sigma)

        if args.store_config == 2:
            # blur_sigma is in px at the render resolution, which is what the parameter
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

        out = get_exposed(out, exposure=exposure, wb_r=wb_r, wb_b=wb_b,
                          vignette=vignette)

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
        if args.store_config == 2:
            json_dict["augment"] = bool(augment)
            json_dict["crop"] = crop
            json_dict["temperature"] = temp
            json_dict["rotate"] = rotate
            json_dict["noise"] = noise

        if args.store_config:
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

    return len(out_array)


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), sys.argv[0])
    parentPath = os.path.dirname(path)
    os.chdir(parentPath)
    run_single_file(get_parser().parse_args(sys.argv[1:]))
