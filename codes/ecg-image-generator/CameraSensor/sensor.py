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
