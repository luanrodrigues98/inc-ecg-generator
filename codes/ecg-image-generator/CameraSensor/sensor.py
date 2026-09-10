import os
import zlib

import cv2
import numpy as np
from linear_light import srgb_to_linear, linear_to_srgb

#Resampling kernel of the downscale. INTER_AREA averages every source pixel that falls
#inside the destination pixel, which is the box integral a photosite performs over its own
#area - the operation this stage exists to model. Every other kernel SAMPLES the source
#with a fixed support instead of integrating it, so on a reduction of 2x or more they read
#a few scattered pixels per output pixel and alias exactly the way the direct render does.
#That is not a contradiction of CRUMPLE_INTERPOLATION in PaperCrumple/crumple.py, which is
#INTER_LANCZOS4: a warp is a same-size resample where there is no support to integrate over
#and sharpness is the only concern.
RESAMPLE_INTERPOLATION = cv2.INTER_AREA

#Largest supersample factor the stage accepts. Not a physical limit - a cost one. The
#render cost grows with the square of the factor and the crumple stage holds several
#full page float32 arrays, so at 300 dpi output a factor of 4 is a 1200 dpi render of
#135 megapixels.
SUPERSAMPLE_MAX = 4

#How far the requested output aspect ratio may sit from the render's before the stage
#refuses. Half a percent is about a pixel and a half on a 300 dpi page, i.e. rounding.
#The bound is tight ON PURPOSE, and it is the one guard in this module that protects a
#physical invariant rather than a budget: see resolve_output_size.
ASPECT_TOLERANCE = 0.005

#Floor of the signal-dependent noise law, as a fraction of full scale. Photon noise goes
#to zero with the signal, but a sensor's read noise does not, so without a floor the black
#trace would come out perfectly clean while the paper around it is grained - the one place
#on this page where a viewer can see the two side by side. 0.02 is the roteiro's figure.
NOISE_FLOOR_FRACTION = 0.02

#Largest sensor_noise the stage accepts, in levels of 0-255. A bound on meaning rather than
#on cost: measured on this page, 20 already delivers a noise_sigma of 9.5 against a corpus
#that runs to 12.36 at its maximum, so 40 is past the noisiest real photograph by a wide
#margin and well into the range where the clipping of the grain against white paper starts
#pulling lum_mean down and competing with the exposure. It is also the roteiro's own
#bracket ceiling for this parameter.
SENSOR_NOISE_MAX = 40.0


def resolve_output_size(render_width,render_height,supersample,output_width,output_height):
    #Decide the pixel size the page is delivered at.
    #Three ways to ask for it, in order of precedence:
    #  neither width nor height   the render divided by the supersample factor, i.e. the
    #                             size the page would have had with no supersampling at
    #                             all. This is the case the batch configs use, and it is
    #                             what keeps --resolution meaning OUTPUT dpi.
    #  one of the two             the other follows from the render's own aspect ratio.
    #  both                       used as given, and checked against that aspect ratio.
    #
    #THE ASPECT RATIO IS NOT FREE, and this is the whole reason the pair is guarded rather
    #than simply obeyed. On ECG paper a vertical millimetre is 0.1 mV and a horizontal
    #millimetre is 40 ms. Scaling the two axes by different factors changes mV per pixel
    #and seconds per pixel by different amounts: the grid stops being square, the trace
    #stops being a faithful plot of the signal, and a digitiser reading the page recovers
    #the wrong amplitude, the wrong duration, or both. A stretched photograph of an ECG is
    #not a photograph of a stretched ECG - the second does not exist.
    #The roteiro asks for (output_width, output_height) pairs sampled jointly from the real
    #corpus. With the aspect ratio pinned there is only one degree of freedom left, so the
    #joint draw collapses to a single number and no joint sampling machinery is needed.
    #The measurement supports the pin: over the 8793 real photographs of the reference
    #corpus the median aspect ratio is 1.2954 against this paper's 11/8.5 = 1.2941. The
    #4:3 families in that corpus are frames with a table and a background around the sheet,
    #which this generator does not render yet.
    aspect = float(render_width)/float(render_height)

    if output_width <= 0 and output_height <= 0:
        width = int(round(render_width/float(supersample)))
        height = int(round(render_height/float(supersample)))
    elif output_height <= 0:
        width = int(output_width)
        height = int(round(width/aspect))
    elif output_width <= 0:
        height = int(output_height)
        width = int(round(height*aspect))
    else:
        width = int(output_width)
        height = int(output_height)
        requested = float(width)/float(height)
        if abs(requested - aspect)/aspect > ASPECT_TOLERANCE:
            raise SystemExit(
                "output_width %d and output_height %d ask for an aspect ratio of %.4f, "
                "and the page renders at %.4f. Scaling the two axes by different factors "
                "changes mV per pixel and seconds per pixel by different amounts, so the "
                "grid stops being square and the trace stops being a faithful plot of the "
                "signal. Set one of the two and let the other follow, or set both to 0 and "
                "let --resolution decide the size."
                % (width,height,requested,aspect))

    if width < 1 or height < 1:
        raise SystemExit(
            "output size resolved to %dx%d, which is not an image. Check output_width, "
            "output_height and supersample." % (width,height))

    if width > render_width or height > render_height:
        raise SystemExit(
            "output size %dx%d is larger than the render at %dx%d. This stage is a "
            "downscale - the sensor integrating an optical image it oversampled - and it "
            "cannot invent detail the render does not have. Raise supersample or "
            "--resolution instead."
            % (width,height,render_width,render_height))

    return width,height


def scale_annotations(json_dict,scale_x,scale_y):
    #Carry the stored geometry down with the pixels.
    #Every coordinate this generator writes is [row, col], both for plotted_pixels
    #(ecg_plot.py) and for the four corners of either bounding box, so the row component
    #takes scale_y and the column component scale_x.
    #NO PADDING OFFSET IS NEEDED HERE, unlike in PaperCrumple.displace_annotations. The
    #annotations live in the frame of the unpadded render while the file may carry a
    #pad_inches border, but a uniform scale of the whole file scales the border with it: a
    #point at unpadded coordinate u sits at u + pad in the file, lands at (u + pad)*s after
    #the resize, and the new border is pad*s, so its unpadded coordinate is exactly u*s.
    #The offset cancels. A non-uniform scale would not cancel, which resolve_output_size
    #refuses for its own reasons.
    #Rewriting json_dict['leads'] through convert_bounding_boxes_to_dict is not an option,
    #for the reason crumple gives: that helper emits integer keys while read_leads, which
    #get_augment calls next, indexes the corners with string keys.
    for lead in json_dict.get('leads',[]):
        for key in ('lead_bounding_box','text_bounding_box'):
            if key not in lead:
                continue
            corners = lead[key]
            for name in list(corners):
                row,col = corners[name]
                corners[name] = [round(float(row)*scale_y,2),round(float(col)*scale_x,2)]
        pixels = lead.get('plotted_pixels')
        if not pixels:
            continue
        lead['plotted_pixels'] = [[round(float(row)*scale_y,2),round(float(col)*scale_x,2)]
                                  for row,col in pixels]

    #The page level figures that are counted in pixels or in pixels per unit of paper.
    #Missing any of these leaves the annotation describing the render while the PNG beside
    #it is the output: the boxes and the trace coordinates would still agree with each
    #other and with the image, so nothing would look wrong, while x_grid would report a
    #grid pitch the file does not have and a digitiser reading millimetres off it would be
    #out by the supersample factor.
    if 'width' in json_dict:
        json_dict['width'] = int(round(json_dict['width']*scale_x))
    if 'height' in json_dict:
        json_dict['height'] = int(round(json_dict['height']*scale_y))
    if 'x_grid' in json_dict:
        json_dict['x_grid'] = round(json_dict['x_grid']*scale_x,3)
    if 'y_grid' in json_dict:
        json_dict['y_grid'] = round(json_dict['y_grid']*scale_y,3)
    if 'resolution' in json_dict:
        #Written by extract_leads at the dpi the page was RENDERED at, which is
        #resolution*supersample. What belongs in the delivered annotation is the dpi the
        #delivered image actually has.
        json_dict['resolution'] = int(round(json_dict['resolution']*scale_x))


#Main function to resample the page onto the sensor grid
def get_resampled(input_file,supersample=1,output_width=0,output_height=0,json_dict=None):
    #The sampling grid of the sensor: the page has been rendered, deformed, defocused and
    #graded at a resolution above the one it is delivered at, and here it is integrated
    #down onto the pixels that are actually kept.
    #This is what makes the supersampling worth doing. A 0.30 mm trace is 1.8 px at 300 dpi
    #and 0.9 px at 150, and the 1 mm minor grid is 1.2 to 2.4 px, so rendering straight at
    #the delivered resolution puts every line of this page at or under the sampling limit
    #and the result depends on where each line happens to fall between pixel centres. A
    #camera does not sample the optical image, it integrates it over each photosite. This
    #stage is that integral.
    #Its position in the chain is fixed by the same physics: after everything the scene and
    #the lens did, and BEFORE the sensor noise, which originates on this grid and must not
    #be downscaled along with the image.
    #A factor of 1 with no explicit output size is a no op: the file is left exactly as it
    #was found, byte for byte. The early return below is what guarantees that - a resize to
    #the same dimensions would still round trip the PNG through imread and imwrite and
    #re-encode it.
    filename = input_file

    #The neutral case, short circuited before the file is even opened. A factor of 1 with
    #no explicit size cannot change the dimensions of anything, and at 300 dpi the page on
    #disk is a 20 MB PNG - decoding it only to compare two numbers and hand it back would
    #cost more than every other guard in this module put together.
    if supersample == 1 and output_width <= 0 and output_height <= 0:
        return filename

    image = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    if image is None:
        return filename
    if image.ndim == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #Read unchanged and put the alpha back untouched, as the other cv2 stages do. The
    #channel count reaching this stage depends on which stages ran before it - get_creased
    #and get_handwritten write three channels, get_crumpled and the bare render four - and
    #a plain cv2.imread would silently drop the fourth.
    render_height,render_width = image.shape[:2]

    width,height = resolve_output_size(render_width,render_height,supersample,
                                       output_width,output_height)
    if width == render_width and height == render_height:
        return filename

    alpha = image[:,:,3] if image.shape[2] == 4 else None
    colour = image[:,:,:3].astype(np.float32)/255.0

    #In linear light. linear_light.py opens by naming resampling first among the operations
    #that are linear on radiance, and this is the resampling stage of the chain. Averaging
    #black against white gives 127 in sRGB and about 188 in linear, and this page is almost
    #entirely a thin black trace on white paper, so the error would land on the signal and
    #on nothing else: the trace would come out heavier and the page darker, which is a gain
    #change in disguise and would put this stage in competition with the exposure for
    #lum_mean. No gain correction follows - an area average conserves energy in linear
    #light exactly, and the small rise it leaves in the DISPLAYED mean is Jensen's
    #inequality over a concave EOTF rather than a gain.
    linear = srgb_to_linear(colour)
    linear = cv2.resize(linear,(width,height),interpolation=RESAMPLE_INTERPOLATION)
    colour = np.clip(linear_to_srgb(linear)*255.0 + 0.5,0,255).astype(np.uint8)

    if alpha is not None:
        #The alpha is constant at 255 on every render this generator produces, so there is
        #no coverage edge here; it is resampled with the same kernel for consistency rather
        #than out of necessity.
        alpha = cv2.resize(alpha,(width,height),interpolation=RESAMPLE_INTERPOLATION)
        image = np.dstack([colour,alpha])
    else:
        image = colour

    cv2.imwrite(filename,image)

    if json_dict is not None:
        scale_annotations(json_dict,
                          float(width)/float(render_width),
                          float(height)/float(render_height))

    return filename


def sensor_noise_level(sensor_noise,sensor_noise_jitter_log2,input_file,seed=-1,start_index=-1):
    #Resolve the noise amplitude this particular sheet was photographed with, in levels of
    #0-255. A jitter of 0 is the deterministic mode and there is no --deterministic_sensor_noise
    #to go with it, for the reason the saturation gives: the neutral value here is 0, so
    #uniform(0, max) cannot express a range with a floor and a jitter around a centre can.
    #
    #The draw is in LOG2 of the amplitude rather than in the amplitude itself, as the
    #exposure jitter is in stops and the contrast and saturation jitters in log2. The
    #corpus is the reason rather than the convention: noise_sigma over the 8793 real
    #photographs runs 2.11 at P05, 3.16 at the median and 9.22 at P95, a right-skewed
    #spread where the distance from the median to P95 is three times the distance down to
    #P05. A symmetric draw on the amplitude would put half its mass in a range the corpus
    #barely occupies; a symmetric draw in log2 follows the shape.
    #
    #A STREAM OF ITS OWN, keyed by the record name, and this is the load-bearing part. The
    #runner consumes the keys of its randomize: block from ONE per-record stream in
    #ALPHABETICAL order, so a key added there shifts every draw that sorts after it -
    #sensor_noise would land between saturation and standard_grid_color and silently
    #re-roll the grid colour, all four trace parameters, the vignette, the white point and
    #the wrinkles of every record in the batch. Drawing here instead costs that stream
    #nothing, which is also why supersample was kept out of it in batch_ptbxl_3000.yaml.
    if sensor_noise_jitter_log2 <= 0:
        return float(sensor_noise)
    record_key = zlib.crc32(os.path.basename(input_file).encode('utf-8'))
    rng = np.random.default_rng([abs(int(seed)),abs(int(start_index)),record_key,3])
    return float(sensor_noise*2.0**rng.uniform(-sensor_noise_jitter_log2,sensor_noise_jitter_log2))

def get_sensor_noise(input_file,sensor_noise=0.0,seed=0,start_index=0):
    #The noise of the sensor, closing stage E-14. Every stage before this one describes
    #something that happened to the page or to the light reaching it; this one is the first
    #that describes the INSTRUMENT, and it is the last thing that happens to the image.
    #
    #POSITION. The roteiro asks for it "at output resolution, after the downscale", and the
    #comment get_resampled carries says the same from the other side. Both are satisfied
    #here, but the call site is further down the chain than that phrasing suggests, and the
    #reason is metrological rather than aesthetic. ImageAugmentation/augment.py runs
    #iaa.Affine(rotate=rot) and iaa.Crop, and BOTH RESAMPLE. Noise injected before them
    #would be smoothed by an angle drawn per image, so the noise_sigma finally measured
    #would be a function of the augment draw rather than of this parameter, and nothing
    #could be calibrated against it. The physical reading agrees: the paper is what is
    #rotated in front of the camera, and the noise of the sensor does not rotate with it.
    #So this runs after get_augment, and after the QR stamp so the stamp is grained like
    #the rest of the page. The JPEG of increment 15 goes after it, and nothing else does.
    #
    #SIGNAL DEPENDENT, not homoscedastic. A photosite's shot noise goes with the square
    #root of the count of photons it collected, so the standard deviation follows the
    #square root of the signal. On this page that is the whole point rather than a
    #refinement: an ECG is a large area of white paper carrying a thin black trace, and
    #uniform noise is visibly wrong in both regions at once - too clean on the paper, too
    #dirty on the trace. The floor of 0.02 keeps the trace from coming out perfectly noise
    #free, which no sensor manages either: read noise survives where photon noise does not.
    #
    #PER CHANNEL, independently. Each photosite carries one colour and counts its own
    #photons, so the three planes are three separate measurements. The visible consequence
    #is chroma speckle rather than clean luminance grain, which is what a real photograph
    #at high ISO shows. It is also what sets the useful range of this parameter: three
    #independent channels combine into the luma the metric reads at about 0.67 of their
    #own sigma, so a given noise_sigma needs a value about half again as large here.
    filename = input_file

    #The neutral case, short circuited before the file is opened, in the idiom get_resampled
    #uses and for the same reason: an imread/imwrite round trip would re-encode the PNG even
    #where every pixel matched, and the regression standard of this branch is byte equality
    #of the finished file at the neutral value.
    if sensor_noise <= 0:
        return filename

    image = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    if image is None:
        return filename
    if image.ndim == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    #A stream of its own, keyed by the record name, in the idiom of PaperCrumple.crumple.
    #The record name is not decoration here: without it every sheet of a batch would carry
    #the SAME field of noise, because the runner hands every record the same seed and the
    #same start_index. A digitiser trained on 4000 images sharing one noise realisation
    #would be learning the realisation.
    record_key = zlib.crc32(os.path.basename(filename).encode('utf-8'))
    rng = np.random.default_rng([abs(int(seed)),abs(int(start_index)),record_key,2])

    alpha = image[:,:,3] if image.shape[2] == 4 else None
    colour = image[:,:,:3].astype(np.float32)

    #In DISPLAY space, and this is the one stage after the downscale that must not convert
    #to linear light - the opposite of what every neighbouring module argues for itself, so
    #it needs saying. The roteiro writes the law in levels of 0-255, and noise_sigma, the
    #feature it is calibrated against, is measured on the delivered sRGB image in those same
    #levels. Converting here would make the parameter mean something other than what the
    #metric reads, and the sqrt law would land on radiance rather than on code values.
    #The honest note is that shot noise is Poisson in the raw domain and this is its
    #display-space approximation; the roteiro specifies the approximation, and it is what
    #keeps the parameter and its target in one unit.
    effective_sigma = sensor_noise*np.sqrt(np.clip(colour/255.0,NOISE_FLOOR_FRACTION,1.0))
    colour = colour + rng.standard_normal(colour.shape,dtype=np.float32)*effective_sigma
    colour = np.clip(colour + 0.5,0,255).astype(np.uint8)

    if alpha is not None:
        #The alpha is left alone. It is constant at 255 on every render this generator
        #produces, and an opacity is not a photon count.
        image = np.dstack([colour,alpha])
    else:
        image = colour

    cv2.imwrite(filename,image)

    #No annotation is touched. This stage changes no geometry, so plotted_pixels, both
    #bounding boxes and the grid pitch all still describe the image - which is why it needs
    #nothing of what scale_annotations does for the downscale.
    return filename
