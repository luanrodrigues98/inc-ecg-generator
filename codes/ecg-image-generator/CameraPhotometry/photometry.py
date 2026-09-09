import cv2
import numpy as np
from linear_light import (srgb_to_linear, linear_to_srgb,
                          GAIN_SAMPLES, GAIN_BRACKET, GAIN_ITERATIONS)

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

#Neutral for the two white balance gains, for the same reason and with the same
#consequence: at 1.0 each the stage must not touch the file at all.
WB_NEUTRAL = 1.0

#D65 in mireds, the reciprocal megakelvin 1e6/T. Colour temperature is drawn on THIS axis
#and not in kelvin, for the same reason exposure is drawn in stops and not in gain: 500 K
#is an enormous shift at 3000 K and invisible at 15000 K, while a mired is roughly the
#same perceived step everywhere. It is also the unit camera filters are sold in.
D65_CCT_K = 6504.0
D65_MIRED = 1.0e6/D65_CCT_K

#Validity range of the CIE D series. The daylight locus is defined from 4000 K up; below
#it the illuminant is a Planckian radiator and the polynomial below is extrapolation, not
#physics. Nothing in the documented parameter range comes near either end - the roteiro
#box of 0.85-1.20 on both gains crosses the locus between about 5500 K and 8000 K - so
#this clamp is a safety net on a caller passing something absurd, not a working limit.
DAYLIGHT_MIN_K = 4000.0
DAYLIGHT_MAX_K = 25000.0

#Linear sRGB primaries from CIE XYZ, the standard matrix. The gains below are the linear
#RGB of an illuminant's white point, which is what a per channel gain in linear light
#physically IS.
_XYZ_TO_LINEAR_SRGB = np.array([[ 3.2406,-1.5372,-0.4986],
                                [-0.9689, 1.8758, 0.0415],
                                [ 0.0557,-0.2040, 1.0570]])

def _locus_gains(cct_k):
    #Linear sRGB of the CIE D series illuminant at this correlated colour temperature,
    #normalised on green. Green is the reference because it is the channel a white balance
    #does not touch: the operation has two degrees of freedom, r/g and b/g, and that is
    #exactly the pair the corpus study reports as wb_r_over_g and wb_b_over_g.
    cct_k = float(min(max(cct_k,DAYLIGHT_MIN_K),DAYLIGHT_MAX_K))
    if cct_k <= 7000.0:
        x = (-4.6070e9/cct_k**3 + 2.9678e6/cct_k**2
             + 0.09911e3/cct_k + 0.244063)
    else:
        x = (-2.0064e9/cct_k**3 + 1.9018e6/cct_k**2
             + 0.24748e3/cct_k + 0.237040)
    y = -3.000*x*x + 2.870*x - 0.275
    xyz = np.array([x/y,1.0,(1.0 - x - y)/y])
    rgb = _XYZ_TO_LINEAR_SRGB.dot(xyz)
    return float(rgb[0]/rgb[1]),float(rgb[2]/rgb[1])

def daylight_gains(mired_offset):
    #(wb_r, wb_b) for an illuminant this many mireds away from D65, ON THE DAYLIGHT LOCUS.
    #This is the whole answer to the roteiro's requirement that the pair be sampled in a
    #CORRELATED way rather than independently: a real illuminant has one degree of freedom,
    #not two, and drawing r and b from separate uniforms produces (high r, high b) pairs -
    #a light that is simultaneously warm and cool - which no lamp, sky or window makes.
    #Sign: a POSITIVE offset is more mireds, therefore FEWER kelvin, therefore WARMER, so
    #wb_r rises and wb_b falls. That is the direction the roteiro states.
    #Exactly (1.0, 1.0) at offset 0, and by construction rather than by rounding: the
    #polynomial evaluated at D65 returns 0.9989 and not 1, and a stage that multiplied by
    #0.9989 at its neutral setting would fail the bit for bit regression every parameter in
    #this chain is required to pass. Dividing by the value at the origin removes both the
    #polynomial's own error and the difference between the D series white point and the
    #sRGB one, which is what makes the neutral exact.
    if mired_offset == 0.0:
        return WB_NEUTRAL,WB_NEUTRAL
    reference_r,reference_b = _locus_gains(D65_CCT_K)
    gain_r,gain_b = _locus_gains(1.0e6/(D65_MIRED + float(mired_offset)))
    return gain_r/reference_r,gain_b/reference_b

def mired_to_cct_k(mired_offset):
    #The kelvin figure for an offset, for the annotation. Kelvin is what a reader
    #recognises; the mired is what the draw is uniform on.
    return 1.0e6/(D65_MIRED + float(mired_offset))

#Luminance weights, BGR to match the channel order cv2 reads a file in. Rec.601, which is
#the convention cv2.cvtColor(..., COLOR_BGR2GRAY) uses and therefore the one lum_mean is
#measured with.
LUMINANCE_WEIGHTS_BGR = np.array([0.114,0.587,0.299],dtype=np.float32)

def _mean_preserving_channel_gain(linear,gains,exposure):
    #Scalar gain that holds the displayed LUMINANCE of the page where it was before the
    #white balance. Same bisection, same constants and same reason as mean_preserving_gain
    #in linear_light.py, which the crumple and the illumination use: without it this stage
    #is a small exposure change in disguise and competes with the exposure parameter for
    #lum_mean, which is the oscillation the calibration notes blame for 85% error.
    #It is written here instead of calling that helper because the helper's argument is a
    #SPATIAL mask and its body decimates the mask alongside the image. A constant gain per
    #channel has no spatial extent to decimate, so the shared function would have to grow a
    #branch that changes how its existing two callers slice their masks - and those two
    #callers are precisely what the bit for bit regression of this increment is measured
    #against. The constants are still imported, so there is one definition of the bracket.
    #
    #Two things differ from that helper, and both were measured rather than assumed.
    #
    #It matches the LUMINANCE WEIGHTED mean and not the flat mean over the three channels.
    #For the crumple and the illumination the two are interchangeable, because their masks
    #are achromatic and scale every channel alike. A white balance does not: at 60 mireds
    #warm the gains are (r 1.277, g 1.0, b 0.675), whose flat mean is 0.984 and whose
    #luminance weighted mean is 1.046. A scalar solved on the flat mean therefore leaves
    #lum_mean 6.3% high by construction - measured at +8.6% with the clipping on top - and
    #hands part of the page brightness back to a parameter that is not supposed to own it.
    #Blue is the channel a warm illuminant cuts hardest and the channel luminance cares
    #least about, so the error is one sided and grows with the warmth.
    #
    #It solves AT THE WORKING EXPOSURE rather than at unit gain. sRGB is not linear, so a
    #pair of images with equal displayed means does not stay equal after both are
    #multiplied by 0.82. Folding the exposure into the solve costs nothing and makes the
    #invariant exact where the image is actually produced. The exposure is still not
    #corrected by this - it multiplies both sides of the comparison, so the gain returned
    #is the one that undoes the WHITE BALANCE alone.
    stride = max(int(max(linear.shape[:2])/GAIN_SAMPLES),1)
    sample = linear[::stride,::stride]*float(exposure)
    target = float((linear_to_srgb(sample)*LUMINANCE_WEIGHTS_BGR).sum(axis=2).mean())
    balanced = sample*gains
    low,high = GAIN_BRACKET
    for _ in range(GAIN_ITERATIONS):
        gain = (low + high)/2.0
        luminance = float((linear_to_srgb(balanced*gain)
                           *LUMINANCE_WEIGHTS_BGR).sum(axis=2).mean())
        if luminance < target:
            low = gain
        else:
            high = gain
    return (low + high)/2.0

#Main function to expose the photograph
def get_exposed(input_file,exposure,wb_r=WB_NEUTRAL,wb_b=WB_NEUTRAL):
    #Exposure of the camera: the scalar gain between the light that reached the sensor and
    #the value recorded for it. It multiplies IN LINEAR LIGHT, which is what makes it a
    #gain at all - the same factor applied to the sRGB values would be a power law on
    #radiance, an operation no camera performs.
    #It runs AFTER the illumination mask, which is the physical order: the room decides
    #how much light falls on each part of the sheet, the camera then decides how much of
    #that it records. It runs BEFORE get_augment, whose gaussian noise belongs to the
    #sensor and to the raw conversion after it.
    #This is the ONLY stage in the chain that is meant to move the page brightness. The
    #crumple and the illumination gradient both undo their effect on the mean with
    #mean_preserving_gain, the blur conserves energy in linear light by construction, and
    #the white balance below corrects its own mean the same way - precisely so that
    #lum_mean has a single owner and a calibration solving for it cannot oscillate between
    #two parameters. Nothing here corrects the mean, and nothing here should.
    #
    #WHITE BALANCE IS FUSED INTO THIS STAGE rather than given a module of its own, and the
    #reason is the architecture: every stage in this generator reads a PNG and writes a
    #PNG, so a separate stage would cost one more uint8 quantisation - half a level - on a
    #page where the whole colour cast being modelled is about ten levels wide. Physically
    #the two are one operation anyway, a diagonal gain matrix in linear light, which is
    #what the roteiro means by "per-channel gains applied alongside exposure".
    filename = input_file
    #A gain of 0 or less is not a dark page, it is a nonsense request: 0 is black and a
    #negative gain has no meaning at all. Returning quietly would make the flag inert for
    #those values, which is the failure this codebase refuses everywhere else - the same
    #reason BLUR_MIN_KERNEL_SIZE exists and the same reason run_batch_from_config rejects
    #an unknown key rather than dropping it.
    if exposure <= 0:
        raise ValueError('exposure must be greater than 0, got %r '
                         '(1.0 is the neutral gain; 0 would be a black page)' % (exposure,))
    if wb_r <= 0 or wb_b <= 0:
        raise ValueError('wb_r and wb_b must be greater than 0, got %r and %r '
                         '(1.0 each is the neutral white balance; 0 would remove a '
                         'channel entirely)' % (wb_r,wb_b))
    if exposure == EXPOSURE_NEUTRAL and wb_r == WB_NEUTRAL and wb_b == WB_NEUTRAL:
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
    linear = srgb_to_linear(colour)

    if wb_r != WB_NEUTRAL or wb_b != WB_NEUTRAL:
        #BGR, because that is the order cv2 read the file in. Green is untouched: a white
        #balance has two degrees of freedom and green is the reference both of them are
        #measured against.
        #The short circuit above matters for more than speed. Normalising a neutral triple
        #would divide by a sum of float literals that is not exactly 1.0, so the stage
        #would multiply by 0.9999998 at its neutral setting and lose the bit for bit
        #regression.
        gains = np.array([wb_b,1.0,wb_r],dtype=np.float32)
        gains = gains*_mean_preserving_channel_gain(linear,gains,exposure)
        linear = linear*gains

    linear = linear*float(exposure)
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
