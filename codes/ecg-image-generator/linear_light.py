import numpy as np

#Shared linear light. Everything a camera does to a scene - resampling, blurring,
#shading, illuminating, exposing - is a linear operation on radiance, and sRGB is not
#radiance. The pair below was written twice before this module existed, once in
#PaperCrumple/crumple.py and once in CameraOptics/optics.py, each with a comment saying
#the shared home belonged to the commit that first needed it in a third place. The
#illumination gradient is that third place, and every photometric stage still to come -
#exposure, white balance, vignette - multiplies in linear light too.

#Bisection that keeps a multiplicative mask from acting as a global gain. The gain is
#solved on a decimated copy: the mean is a global statistic and does not need every
#pixel.
GAIN_SAMPLES = 300
GAIN_BRACKET = (0.5,4.0)
GAIN_ITERATIONS = 24

def srgb_to_linear(image):
    #sRGB EOTF, exact piecewise form.
    return np.where(image <= 0.04045,
                    image/12.92,
                    np.power((image + 0.055)/1.055,2.4)).astype(np.float32)

def linear_to_srgb(image):
    #Inverse of srgb_to_linear.
    image = np.clip(image,0.0,1.0)
    return np.where(image <= 0.0031308,
                    image*12.92,
                    1.055*np.power(image,1.0/2.4) - 0.055).astype(np.float32)

def mean_preserving_gain(linear,mask,reference):
    #Scalar gain that holds the displayed mean of the page where it was before the mask.
    #A mask normalised to mean 1.0 is only gain neutral on a signal with headroom above
    #and below. Here about 62% of the page is paper rendered at pure white: the parts the
    #mask brightens clip and the parts it darkens darken freely, so a mean 1.0 mask still
    #darkens the page - measured at -3.6% for the first choice of crumple slope. Left
    #uncorrected the stage is a small exposure change in disguise and competes with the
    #exposure parameter, which is the oscillation the calibration notes warn about.
    #The mean is matched in DISPLAY space, which is where lum_mean is measured, and
    #against a reference image the caller chooses: the crumple hands in the page as it was
    #BEFORE its deformation, since resampling a trace this thin lightens it slightly and
    #that belongs to the stage too, while a stage that moves no geometry hands in the same
    #image it is about to multiply.
    stride = max(int(max(linear.shape[:2])/GAIN_SAMPLES),1)
    sample = linear[::stride,::stride]
    target = float(linear_to_srgb(reference[::stride,::stride]).mean())
    masked = sample*mask[::stride,::stride,np.newaxis]
    low,high = GAIN_BRACKET
    for _ in range(GAIN_ITERATIONS):
        gain = (low + high)/2.0
        if float(linear_to_srgb(masked*gain).mean()) < target:
            low = gain
        else:
            high = gain
    return (low + high)/2.0
