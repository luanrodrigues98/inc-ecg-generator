import cv2
import numpy as np
from linear_light import srgb_to_linear, linear_to_srgb, mean_preserving_gain

#Distance of the light source from the centre of the sheet, measured along the azimuth in
#units of half the page diagonal. It is the constant that puts the QUADRATIC term in the
#mask, and the quadratic term is not decoration: a purely linear gradient is
#antisymmetric about the centre, so the border ring averages its bright and dim halves
#back to the centre value and lum_center_edge_diff - the metric this parameter is
#calibrated against - does not move at all. What survives that averaging is the
#CURVATURE, and the curvature is what an inverse square falloff from a nearby source
#has and a linear ramp does not.
#Expanding 1/(D - t)^2 about the centre gives 1 + 2t/D + 3t^2/D^2 + ..., so the weight of
#the quadratic term relative to the linear one is 1.5/D: this one number is the whole
#trade. Small D concentrates the falloff near the source; large D is the forbidden pure
#ramp, whose curvature - and therefore whose lum_center_edge_diff response - goes to zero.
#D 2.5 gives the most concentrated falloff that still lights the page rather than one
#corner of it: measured over a real render it carries the strongest curvature of the
#usable range (c-e of the mask -0.082, against -0.048 at D 4 and -0.019 at D 10) while
#the visible contrast stays within 6% of the best value any D reaches. Going lower does
#NOT concentrate the light further - it flattens the page, because the normalisation
#below divides by a statistic that a sharper peak inflates. The selection tables are in
#out/illum_eval/RESULTADOS.md.
ILLUM_SOURCE_DISTANCE = 2.5

#Percentile of the field used to scale it, instead of its maximum. This is the constant
#that decides whether the stage is VISIBLE at all, and it is worth being explicit about
#why. About 43% of this page is paper rendered at pure white, so the half of the mask
#that brightens has nowhere to go: everything the eye can see is the half that darkens.
#Dividing by the MAXIMUM - the obvious choice, and the first one tried here - hands the
#whole budget to the single brightest pixel, which is exactly the one that cannot use it,
#and the far field then falls by only a few percent. Measured at illum_strength 0.5 the
#page carried 11.9 levels of visible contrast that way, which reads as no light at all.
#Scaling on the 90th percentile instead lets the near corner overshoot into the clipping
#it was going to hit anyway and spends the range on the far field, which can show it:
#18.8 levels, and the mask floor drops from 0.74 to 0.49.
#The cost is that illum_strength is no longer the gain at the brightest POINT of the
#page; it is the gain at the 90th percentile of the falloff, and the peak runs past it.
#Useful range 0 - 1.3, wider than the 0 - 0.5 the roteiro suggests: with the exponential
#mask the top of that wider range still holds the page mean to 0.1% and crushes nothing.
ILLUM_NORMALISE_PERCENTILE = 90.0

#Floor on the mask, mirroring CRUMPLE_MIN_LAMBERT. With the exponential form below the
#mask is positive by construction and this never binds inside the documented range: it
#would take an illum_strength near 3 to reach it. Kept as a safety net only.
ILLUM_MIN_GAIN = 0.05

def illumination_mask(shape,illum_strength,illum_azimuth_deg):
    #Multiplicative gain per pixel, mean 1.0.
    height,width = shape
    rows,cols = np.mgrid[0:height,0:width]
    #Normalised by HALF THE DIAGONAL, so the mask is invariant to the page aspect ratio
    #and to the render resolution, and so that the reach of the field is one unit at the
    #corners whatever the padding the caller added.
    half_y = (height - 1)/2.0
    half_x = (width - 1)/2.0
    half_diagonal = float(np.hypot(half_x,half_y))
    if half_diagonal <= 0:
        return None
    offset_y = (rows - half_y)/half_diagonal
    offset_x = (cols - half_x)/half_diagonal

    #Distance towards the source, projected on the azimuth. The image y axis points down,
    #which is the same convention the crumple uses to build its light vector, so an
    #azimuth means the same direction in both stages.
    azimuth = np.radians(illum_azimuth_deg)
    towards = offset_x*np.cos(azimuth) + offset_y*np.sin(azimuth)

    #Irradiance of a point source: inverse square of the distance to it.
    field = 1.0/np.square(ILLUM_SOURCE_DISTANCE - towards)
    field = field - field.mean()
    #Scaled on a high percentile rather than on the maximum, for the reason set out at
    #ILLUM_NORMALISE_PERCENTILE: the maximum sits in the clipped highlights, where no
    #gain is visible, so normalising there spends the parameter on nothing. The field is
    #deliberately NOT clipped afterwards - the near corner is allowed past 1 + strength,
    #which is what a light source close to the sheet actually does.
    scale = float(np.percentile(np.abs(field),ILLUM_NORMALISE_PERCENTILE))
    if scale <= 0:
        return None
    field = field/scale

    #Exponential rather than 1 + strength*field, which is to say illum_strength is a
    #number of STOPS at the 90th percentile of the falloff and not an additive fraction.
    #The two agree to first order, so the bottom of the range behaves identically; they
    #part company at the top, and that is the whole point. The additive form crosses zero
    #at illum_strength 0.98 and then has to be clamped: measured at 1.3 it flattened 11%
    #of the sheet onto the floor as one crushed black plateau, destroyed the trace inside
    #it, and dragged the page mean down 5.5% because the mean preserving bisection ran out
    #of bracket trying to undo it. The exponential cannot reach zero at all - at 1.3 the
    #far corner sits at 0.15 of the reference, dark but still paper with a trace on it -
    #and the page mean stays within 0.1%. Light attenuates multiplicatively; expressing it
    #that way is also the more honest model.
    mask = np.exp(illum_strength*field)
    mask = np.maximum(mask,ILLUM_MIN_GAIN)
    #Mean 1.0, which the roteiro requires of every multiplicative mask in the chain: an
    #un-normalised gradient carries a brightness change with it and then competes with the
    #exposure parameter for lum_mean, which is how a previous calibration oscillated to
    #85% error.
    return (mask/float(mask.mean())).astype(np.float32)

#Main function to light the sheet from one side
def get_illuminated(input_file,illum_strength,illum_azimuth_deg):
    #Non-uniform illumination of the scene: window light, or a lamp off to one side. It is
    #the first PHOTOMETRIC stage - the scene is already formed and already defocused, and
    #what follows it is the sensor: the gaussian noise and the colour temperature of
    #get_augment.
    #The azimuth is not this stage's to choose. It is resolved once per image by
    #light_azimuth in PaperCrumple/crumple.py and handed to the crumple as well, so the
    #shadow on the far side of a fold and the dim end of the page agree on where the light
    #is. Two light directions in one photograph is impossible and plainly visible.
    #illum_strength 0 is a no op: the file is left exactly as it was found.
    filename = input_file
    if illum_strength <= 0:
        return filename

    image = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    if image is None:
        return filename
    if image.ndim == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    #Read unchanged and put the alpha back untouched, as the blur does: the channel count
    #reaching this stage depends on which stages ran before it, and a plain cv2.imread
    #would silently drop the fourth channel.
    alpha = image[:,:,3] if image.shape[2] == 4 else None
    colour = image[:,:,:3].astype(np.float32)/255.0

    mask = illumination_mask(colour.shape[:2],illum_strength,illum_azimuth_deg)
    if mask is None:
        return filename

    #In linear light, which is the only space in which lighting a surface is a
    #multiplication at all.
    linear = srgb_to_linear(colour)
    #A mean 1.0 mask is still not gain neutral on this page, because most of it is paper
    #at pure white: the brightened side clips and the darkened side does not, so the
    #displayed mean drops. The gain puts it back, which is what keeps this parameter from
    #being an exposure change in disguise. The reference is the image itself - unlike the
    #crumple, this stage moves no pixels, so there is nothing else it could be.
    linear = linear*mask[:,:,np.newaxis]*mean_preserving_gain(linear,mask,linear)
    colour = np.clip(linear_to_srgb(linear)*255.0 + 0.5,0,255).astype(np.uint8)

    if alpha is not None:
        image = np.dstack([colour,alpha])
    else:
        image = colour

    cv2.imwrite(filename,image)

    #json_dict is deliberately not taken, for the same reason the blur does not take it: a
    #multiply moves no geometry, so the lead bounding boxes, the text bounding boxes and
    #plotted_pixels all stay valid as they are.
    return filename
