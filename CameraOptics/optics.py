import cv2
import numpy as np
from linear_light import srgb_to_linear, linear_to_srgb

#Half width of the kernel, as a multiple of sigma. Three sigma holds 99.7% of the
#gaussian; truncating closer leaves a step at the edge of the kernel, which on a page of
#thin black lines on white shows up as a faint halo at the truncation radius.
BLUR_KERNEL_SIGMAS = 3.0

#Smallest kernel the stage will use. cv2 treats a ksize of 1 as the identity along that
#axis, so without a floor a sigma below 1/6 would parse without error and change nothing -
#the class of silently inert flag that this generator already has too many of.
BLUR_MIN_KERNEL_SIZE = 3

#Border handling. The render carries the black axes frame matplotlib draws around the
#page at its outermost pixels; BORDER_REFLECT_101, the cv2 default, would mirror that
#frame back into the paper and lay a dark line just inside the edge. Replication keeps the
#frame where it is and lets it blur outwards like everything else.
BLUR_BORDER_MODE = cv2.BORDER_REPLICATE

def kernel_size(blur_sigma):
    #Odd by construction, since cv2 requires it.
    size = 2*int(round(BLUR_KERNEL_SIGMAS*blur_sigma)) + 1
    return max(size,BLUR_MIN_KERNEL_SIZE)

#Main function to defocus the photograph
def get_blurred(input_file,blur_sigma):
    #Optical blur of an imperfectly focused camera. It acts on the scene as already
    #formed - the deformed sheet - and before anything photometric or sensor side. The
    #gaussian noise of get_augment therefore lands on top of the blur, which is the way
    #round a camera does it; blurring after the noise would smooth the noise away.
    #blur_sigma is in PIXELS AT THE RENDER RESOLUTION, not in mm of paper. Trace thickness
    #and crumple scale are paper lengths because a stylus and a fold are properties of the
    #sheet; defocus is a property of the camera and is measured on its sensor. The
    #consequence is that this stage, alone among the ones added so far, is NOT invariant to
    #--resolution: the same sigma is a different blur in mm at 150 and at 300 dpi.
    #blur_sigma 0 is a no op: the file is left exactly as it was found.
    filename = input_file
    if blur_sigma <= 0:
        return filename

    image = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    if image is None:
        return filename
    if image.ndim == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #Read unchanged and put the alpha back untouched. The channel count reaching this
    #stage depends on which stages ran before it - get_creased and get_handwritten write
    #three channels, get_crumpled and the bare render four - and a plain cv2.imread would
    #silently drop the fourth and hand get_augment a different image than it used to get.
    #The alpha is constant at 255 on every render this generator produces, so there is no
    #coverage edge to soften and no need to premultiply.
    alpha = image[:,:,3] if image.shape[2] == 4 else None
    colour = image[:,:,:3].astype(np.float32)/255.0

    #In linear light. Blurring in sRGB darkens high contrast edges, and this page is almost
    #entirely black trace on white paper, so the error would land exactly on the signal.
    #No gain correction follows: a blur conserves energy in linear light exactly, and the
    #small rise it produces in the DISPLAYED mean is Jensen's inequality over a concave
    #EOTF, not a gain. Correcting it would make this stage compete with the exposure
    #parameter for lum_mean, which is the oscillation the calibration notes warn about.
    size = kernel_size(blur_sigma)
    linear = srgb_to_linear(colour)
    linear = cv2.GaussianBlur(linear,(size,size),sigmaX=blur_sigma,sigmaY=blur_sigma,
                              borderType=BLUR_BORDER_MODE)
    colour = np.clip(linear_to_srgb(linear)*255.0 + 0.5,0,255).astype(np.uint8)

    if alpha is not None:
        image = np.dstack([colour,alpha])
    else:
        image = colour

    cv2.imwrite(filename,image)

    #json_dict is deliberately not taken: a blur moves no geometry, so the lead bounding
    #boxes, the text bounding boxes and plotted_pixels all stay valid as they are.
    return filename
