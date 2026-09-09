import cv2
import numpy as np
from linear_light import srgb_to_linear, linear_to_srgb

#Gain that leaves the page as it was found. Exposure is the first parameter in this chain
#whose neutral value is not zero, because it is a MULTIPLIER and not an amount of
#something added: a stylus dropout, a fold, a defocus and a lighting gradient all vanish
#at 0, while a gain vanishes at 1. The stage returns without opening the file at this
#value rather than multiplying by 1.0 and writing the result back, because the round trip
#is not a guaranteed per pixel identity - srgb_to_linear and linear_to_srgb are a
#floating point pair and the final quantisation adds half a level before truncating - and
#the roteiro asks every new parameter to reproduce the previous output bit for bit at its
#neutral value.
EXPOSURE_NEUTRAL = 1.0

#Main function to expose the photograph
def get_exposed(input_file,exposure):
    #Exposure of the camera: the scalar gain between the light that reached the sensor and
    #the value recorded for it. It multiplies IN LINEAR LIGHT, which is what makes it a
    #gain at all - the same factor applied to the sRGB values would be a power law on
    #radiance, an operation no camera performs.
    #It runs AFTER the illumination mask, which is the physical order: the room decides
    #how much light falls on each part of the sheet, the camera then decides how much of
    #that it records. It runs BEFORE get_augment, whose gaussian noise and colour
    #temperature belong to the sensor and to the raw conversion after it.
    #This is the ONLY stage in the chain that is meant to move the page brightness. The
    #crumple and the illumination gradient both undo their effect on the mean with
    #mean_preserving_gain, and the blur conserves energy in linear light by construction,
    #precisely so that lum_mean has a single owner and a calibration solving for it cannot
    #oscillate between two parameters. Nothing here corrects the mean, and nothing here
    #should.
    filename = input_file
    #A gain of 0 or less is not a dark page, it is a nonsense request: 0 is black and a
    #negative gain has no meaning at all. Returning quietly would make the flag inert for
    #those values, which is the failure this codebase refuses everywhere else - the same
    #reason BLUR_MIN_KERNEL_SIZE exists and the same reason run_batch_from_config rejects
    #an unknown key rather than dropping it.
    if exposure <= 0:
        raise ValueError('exposure must be greater than 0, got %r '
                         '(1.0 is the neutral gain; 0 would be a black page)' % (exposure,))
    if exposure == EXPOSURE_NEUTRAL:
        return filename

    image = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    if image is None:
        return filename
    if image.ndim == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #Read unchanged and put the alpha back untouched, as the blur and the illumination do:
    #the channel count reaching this stage depends on which stages ran before it -
    #get_creased and get_handwritten write three channels, get_crumpled and the bare render
    #four - and a plain cv2.imread would silently drop the fourth.
    alpha = image[:,:,3] if image.shape[2] == 4 else None
    colour = image[:,:,:3].astype(np.float32)/255.0

    #The roteiro asks this stage to stay in float and leave clipping to the black and
    #white point increment. That is not available here and the reason is architectural,
    #not a shortcut: every stage in this generator reads a PNG and writes a PNG, so the
    #image is quantised to 8 bits and clipped at each boundary whatever this function
    #does. Carrying the photometric chain in float would mean fusing exposure, white
    #balance, vignette, contrast and the clipping points into one stage - a rewrite of the
    #chain rather than one parameter.
    #What the roteiro was protecting is still intact: clip_highlights_pct rises with the
    #exposure here and stays calibratable. What is lost is only the headroom ABOVE white
    #that a float chain would keep, and this page has very little of it to lose - roughly
    #43% of it is already paper at pure white in the bare render, before any photometric
    #stage runs at all.
    linear = srgb_to_linear(colour)*float(exposure)
    colour = np.clip(linear_to_srgb(linear)*255.0 + 0.5,0,255).astype(np.uint8)

    if alpha is not None:
        image = np.dstack([colour,alpha])
    else:
        image = colour

    cv2.imwrite(filename,image)

    #json_dict is deliberately not taken, for the same reason the blur and the illumination
    #do not take it: a multiply moves no geometry, so the lead bounding boxes, the text
    #bounding boxes and plotted_pixels all stay valid as they are.
    return filename
