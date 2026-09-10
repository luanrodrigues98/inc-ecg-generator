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

#Neutral for the vignette, and it goes back to ZERO after two parameters whose neutral was
#one. The header above makes a point of exposure being the first parameter in this chain
#whose neutral is not zero, because it is a gain rather than an amount of something added.
#The vignette is neither: it is the STRENGTH of a falloff, 1 - v*r^2 is the identity mask
#at v = 0, and the neutral is zero again. The consequence is the one every parameter here
#carries - at the neutral value the stage must not touch the file at all.
VIGNETTE_NEUTRAL = 0.0

#Floor on the mask, mirroring ILLUM_MIN_GAIN and CRUMPLE_MIN_LAMBERT. 1 - v*r^2 takes its
#minimum 1 - v at the corners, so the floor only binds above v 0.95: it is unreachable
#inside the documented range of 0 - 0.6 and inside the roteiro's 0 - 0.9 calibration
#bracket. That headroom is deliberate rather than merely cautious. The closed form
#mask.mean() = 1 - v/3 below is used as a self test of the normalisation, and it is exact
#only while the floor does not bind - a higher floor would make the test quietly stop
#being a test at the top of the range, which is where it is most needed.
VIGNETTE_MIN_GAIN = 0.05

#Largest strength the mask accepts, exclusive. At exactly 1 the corners are zero before
#normalisation, which is a black ring and not a vignette, and the roteiro's own bracket
#stops at 0.9.
VIGNETTE_MAX = 1.0

#Neutral for the tone curve, and it is a MULTIPLIER again after the vignette went back to
#zero. Contrast scales the DISTANCE of every pixel from an anchor, so 1.0 leaves that
#distance alone and 0 would collapse the page onto a single value. The consequence is the
#one every parameter in this chain carries - at the neutral value the stage must not touch
#the file at all.
CONTRAST_NEUTRAL = 1.0

#The anchor of the curve: mid grey, exactly as the roteiro writes it. It stays at 0.5
#rather than being moved to the page mean, and the mean is restored by the offset solved
#in _mean_preserving_offset instead. The two are the SAME CURVE - a + (in - a)*c equals
#0.5 + (in - 0.5)*c + d with d = (a - 0.5)*(1 - c) - but only the offset form can be
#solved, for the reason given there.
CONTRAST_ANCHOR = 0.5

#Half width of the bracket the offset is solved in, placed AROUND THE CLOSED FORM rather
#than fixed at the origin the way GAIN_BRACKET is. The root moves with both the contrast
#and the page mean - at contrast 4.0, the top of the roteiro's calibration bracket, on a
#page of mean 0.9 it sits at -1.2 - so a fixed interval covering that would have to be
#enormous, and a bisection in a needlessly wide bracket spends its iterations on the part
#of the range it already knows is wrong. Centred on the closed form, this half width is
#the room the CLIPPING needs, which is the only thing the closed form does not account
#for.
CONTRAST_OFFSET_HALF_WIDTH = 0.5

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

def vignette_mask(shape,vignette):
    #Multiplicative gain per pixel, mean 1.0. Radial luminance falloff of the optical
    #system: every lens delivers less light off axis, and the roteiro keeps it apart from
    #the illumination gradient of the stage before because the two have different physical
    #causes and different SHAPES. The gradient is asymmetric - it points at the lamp - and
    #this one is radially symmetric, a property of the camera and not of the room. That
    #difference is also the only honest way to tell them apart in the measurements, since
    #they move the same spatial statistics; see the edge band table in the evaluation.
    height,width = shape
    rows,cols = np.mgrid[0:height,0:width]
    #Normalised by HALF THE DIAGONAL, exactly as illumination_mask does and exactly as the
    #roteiro requires. Dividing x and y separately - which is what the one published
    #implementation of this transform does - gives an ELLIPTICAL mask whose shape follows
    #the page aspect ratio, so one value of the parameter would mean different things on a
    #3:2 page and on a square one. By the diagonal the mask is circular, r is exactly 1 at
    #the corners whatever the padding, and the parameter is invariant to both the aspect
    #ratio and the render resolution.
    half_y = (height - 1)/2.0
    half_x = (width - 1)/2.0
    half_diagonal = float(np.hypot(half_x,half_y))
    if half_diagonal <= 0:
        return None
    offset_y = (rows - half_y)/half_diagonal
    offset_x = (cols - half_x)/half_diagonal
    radius_squared = offset_x*offset_x + offset_y*offset_y

    #The roteiro's additive form, kept literally rather than replaced by the exponential
    #the illumination gradient ended up needing. The two stages fail differently and the
    #difference is why the deviation made there is not made here: the illumination field is
    #antisymmetric about the centre, so 1 + s*field crossed zero in the MIDDLE of its
    #useful range and had to be clamped. Here the minimum is 1 - v, reached only at the
    #corners, and the whole documented range stays clear of it. The floor is the safety net
    #for a caller that ignores the bound, not a working limit.
    mask = np.maximum(1.0 - float(vignette)*radius_squared,VIGNETTE_MIN_GAIN)

    #Mean 1.0, which the roteiro calls MANDATORY for this parameter by name: an
    #un-normalised falloff is a brightness change wearing a mask, and it would then compete
    #with exposure for lum_mean - the oscillation the calibration notes blame for 85%
    #median error. It is necessary and NOT sufficient on this page, for the reason set out
    #where the bisection is called below.
    #The divisor has a closed form worth knowing, and worth checking against: under this
    #normalisation E[r^2] = 1/3 exactly for any rectangle, independent of the aspect ratio,
    #because E[x^2] = half_x^2/3 and E[y^2] = half_y^2/3 sum to a third of the squared half
    #diagonal. So mask.mean() = 1 - v/3, the normalised centre is 1/(1 - v/3) and the
    #normalised corner is (1 - v)/(1 - v/3): 1.25 and 0.50 at v 0.6, 1.43 and 0.14 at 0.9.
    return (mask/float(mask.mean())).astype(np.float32)

#Luminance weights, BGR to match the channel order cv2 reads a file in. Rec.601, which is
#the convention cv2.cvtColor(..., COLOR_BGR2GRAY) uses and therefore the one lum_mean is
#measured with.
LUMINANCE_WEIGHTS_BGR = np.array([0.114,0.587,0.299],dtype=np.float32)

def _mean_preserving_gain(linear,gains,exposure,mask=None):
    #Scalar gain that holds the displayed LUMINANCE of the page where it was before the
    #white balance and the vignette. Same bisection, same constants and same reason as
    #mean_preserving_gain
    #in linear_light.py, which the crumple and the illumination use: without it this stage
    #is a small exposure change in disguise and competes with the exposure parameter for
    #lum_mean, which is the oscillation the calibration notes blame for 85% error.
    #It is written here instead of calling that helper because of the two differences set
    #out below, and those two differences are also why the vignette solves HERE rather than
    #calling the shared helper the way the illumination gradient does. The constants are
    #still imported, so there is one definition of the bracket.
    #
    #ONE SOLVE COVERS THE WHOLE FUSED MULTIPLIER - the channel gains of the white balance
    #and the spatial mask of the vignette together. Two bisections run in sequence would
    #each be exact only with the other absent, since the invariant they hold is measured in
    #display space and the sRGB curve does not distribute over the two multiplies. One
    #scalar solved against the finished product is exact, and it is also the honest shape
    #of the requirement: what must not move is lum_mean of the image this stage writes, not
    #lum_mean of an intermediate that never reaches a file.
    #The mask is decimated on the SAME stride as the image, as mean_preserving_gain does,
    #and the multiply sits inside a branch so that the white balance only path keeps the
    #exact floating point operations it had before the vignette existed - which is what the
    #bit for bit regression of this increment is measured against.
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
    if mask is not None:
        balanced = balanced*mask[::stride,::stride,np.newaxis]
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

def _mean_preserving_offset(display,contrast):
    #Additive offset in DISPLAY space that holds the luminance weighted mean of the page
    #across the tone curve. Same bisection, same constants and the same reason as
    #_mean_preserving_gain above: without it the contrast is a brightness change in
    #disguise and competes with the exposure for lum_mean, which is the oscillation the
    #calibration notes blame for 85% median error.
    #
    #WHY THIS STAGE NEEDS ONE AT ALL, WHEN THE ROTEIRO SAYS IT DOES NOT. The roteiro
    #anchors the curve at mid grey and argues that anchoring anywhere else shifts lum_mean
    #and breaks the already calibrated exposure. That is true of an image whose mean IS mid
    #grey. This page is most of the way to white - measured 0.857 in display space at
    #exposure 0.82, since about 62% of it is paper - so the mid grey anchor moves lum_mean
    #by -14.7% at contrast 0.6 and +16.7% at 1.8, and at 1.8 it drives 68% of the sheet
    #into pure white and takes the millimetre grid with it. The roteiro's own acceptance
    #criterion for this same parameter is 2%. The two halves of that section cannot both
    #hold on an ECG page, and it is the anchor that does not survive the contact: solving
    #a + (in - a)*c for a constant mean gives a = mean, in closed form. The curve stays
    #literal and the offset below is what puts the mean back.
    #
    #THE OFFSET IS THE SOLVED VARIABLE, NOT THE ANCHOR, and that is not a matter of taste.
    #d out/d a = 1 - c, which CHANGES SIGN at c = 1, so a bisection on the anchor would run
    #backwards over half the documented range. d out/d offset = +1 everywhere, so the loop
    #below keeps exactly the comparison the other bisections in this codebase use.
    #
    #SEQUENTIAL WITH _mean_preserving_gain AND STILL EXACT, which the white balance and the
    #vignette were not - those two had to share a single solve. The difference is that they
    #are multiplies inside the same nonlinearity, so neither is exact with the other
    #present, while this one acts on the finished display array and matches the mean of
    #that same array. The two invariants compose, and exposure stays the single owner of
    #lum_mean.
    stride = max(int(max(display.shape[:2])/GAIN_SAMPLES),1)
    sample = display[::stride,::stride]
    target = float((sample*LUMINANCE_WEIGHTS_BGR).sum(axis=2).mean())

    #Closed form, exact wherever the curve clips nothing, obtained by setting the mean of
    #0.5 + (in - 0.5)*c + d equal to the mean of in. It is the bracket's centre and it is
    #also the analytic self test of this increment - the role 1 - v/3 plays for the
    #vignette mask. Where it disagrees with the solved value, the difference IS the
    #clipping, and that is the figure the evaluation reports.
    offset = (1.0 - float(contrast))*(target - CONTRAST_ANCHOR)
    low = offset - CONTRAST_OFFSET_HALF_WIDTH
    high = offset + CONTRAST_OFFSET_HALF_WIDTH

    #The curve without the offset, computed once outside the loop: the bisection only
    #moves the constant added to it.
    curve = CONTRAST_ANCHOR + (sample - CONTRAST_ANCHOR)*float(contrast)
    for _ in range(GAIN_ITERATIONS):
        offset = (low + high)/2.0
        #Clipped inside the loop, because the clip is the whole reason the closed form is
        #not the answer. Matching [0,1] here is matching what the uint8 cast does to the
        #full image afterwards.
        shifted = np.clip(curve + offset,0.0,1.0)
        if float((shifted*LUMINANCE_WEIGHTS_BGR).sum(axis=2).mean()) < target:
            low = offset
        else:
            high = offset
    return (low + high)/2.0

#Main function to expose the photograph
def get_exposed(input_file,exposure,wb_r=WB_NEUTRAL,wb_b=WB_NEUTRAL,
                vignette=VIGNETTE_NEUTRAL,contrast=CONTRAST_NEUTRAL):
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
    #
    #THE VIGNETTE IS FUSED HERE TOO, and for a stronger reason than the quantisation one.
    #linear_to_srgb clips to [0,1], so every stage boundary is a clip as well as a
    #quantisation, and after the exposure roughly half of this page sits at pure white. A
    #vignette applied as a separate stage AFTER this one would therefore darken pixels that
    #have already been flattened onto 255: the corner it dims would come back as a flat
    #grey with the grid and the trace crushed out of it, because the values that
    #distinguished them were discarded by the earlier clip. Multiplied in here, the mask and
    #the exposure meet the ceiling ONCE, and the corner the vignette darkens keeps the
    #detail it had. The roteiro asks the photometric chain to stay in float for exactly this
    #reason and the comment further down explains why it cannot here; fusing the vignette is
    #the part of that ask this branch can actually honour.
    #The order among the three is irrelevant to the arithmetic - they are all
    #multiplications in linear light and they commute - but not to the mean preserving
    #solve, which is nonlinear, and that is why one bisection covers all of them.
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
    #A negative strength is an INVERTED vignette - a bright ring on a dark centre, which no
    #lens produces - and at 1 the corners reach zero before the mask is normalised. Refused
    #rather than clamped, for the reason the exposure guard above gives.
    if vignette < VIGNETTE_NEUTRAL or vignette >= VIGNETTE_MAX:
        raise ValueError('vignette must be in [0, 1), got %r '
                         '(0 is the neutral falloff; at 1 the corners are black before '
                         'the mask is normalised)' % (vignette,))
    #A contrast of 0 collapses the page onto a single value and a negative one INVERTS it,
    #into a white trace on dark paper. Neither is a tone curve. Refused rather than clamped,
    #for the reason the exposure guard above gives. There is no upper bound, exactly as
    #there is none on the exposure: a high contrast is a hard curve, not a meaningless one,
    #and the roteiro's own calibration bracket for it runs to 4.0.
    if contrast <= 0:
        raise ValueError('contrast must be greater than 0, got %r '
                         '(1.0 is the neutral curve; 0 would flatten the page onto a '
                         'single value)' % (contrast,))
    #The last two terms are not decoration. Without them --vignette and --contrast would be
    #silently inert whenever the parameters before them sit at their neutrals, which is a
    #flag that parses and does nothing - the upstream failure this codebase refuses
    #everywhere else.
    if (exposure == EXPOSURE_NEUTRAL and wb_r == WB_NEUTRAL and wb_b == WB_NEUTRAL
            and vignette == VIGNETTE_NEUTRAL and contrast == CONTRAST_NEUTRAL):
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

    #None rather than an identity mask when the parameter is off, so that the white balance
    #only path below reaches the bisection with exactly the arguments it had before this
    #parameter existed. An all ones array would be a per pixel identity in exact
    #arithmetic, but it would also change which expressions are evaluated, and the
    #regression this increment is measured against is byte equality of the finished PNG.
    mask = (vignette_mask(linear.shape[:2],vignette)
            if vignette != VIGNETTE_NEUTRAL else None)

    if wb_r != WB_NEUTRAL or wb_b != WB_NEUTRAL or mask is not None:
        #BGR, because that is the order cv2 read the file in. Green is untouched: a white
        #balance has two degrees of freedom and green is the reference both of them are
        #measured against.
        #The short circuit above matters for more than speed. Normalising a neutral triple
        #would divide by a sum of float literals that is not exactly 1.0, so the stage
        #would multiply by 0.9999998 at its neutral setting and lose the bit for bit
        #regression.
        #With the white balance neutral the triple is (1, 1, 1) and this reduces to the
        #scalar the vignette needs, which is why one branch serves both parameters.
        gains = np.array([wb_b,1.0,wb_r],dtype=np.float32)
        gains = gains*_mean_preserving_gain(linear,gains,exposure,mask)
        linear = linear*gains
        if mask is not None:
            #A mask of mean 1.0 is still not gain neutral on this page, which is why the
            #roteiro's mandatory normalisation is necessary and not sufficient, and why the
            #bisection above exists. Normalising to mean 1.0 pushes the CENTRE up by
            #1/(1 - v/3) - 25% at v 0.6 - into paper that is already at pure white and can
            #do nothing with it, while the corners darken freely. The displayed mean
            #therefore falls, and the scalar folded into gains puts it back. The
            #illumination gradient carries the same asymmetry in mirror image and solves it
            #the same way.
            linear = linear*mask[:,:,np.newaxis]

    linear = linear*float(exposure)

    #The tone curve, and it is the first thing in this stage that is NOT a multiply in
    #linear light. It belongs in DISPLAY space, which is where the roteiro puts it and
    #where contrast_rms is measured, so it lands between the sRGB conversion and the
    #quantisation. That makes the roteiro's "after the sRGB conversion, before clipping"
    #literal here rather than aspirational: the clip folded into the uint8 cast on the last
    #line is the only one left after it.
    #Being nonlinear is also why it does not join the fused multiplier above and why its
    #mean is restored by a second, sequential solve - see _mean_preserving_offset.
    display = linear_to_srgb(linear)
    if contrast != CONTRAST_NEUTRAL:
        display = (CONTRAST_ANCHOR + (display - CONTRAST_ANCHOR)*float(contrast)
                   + _mean_preserving_offset(display,contrast))
    colour = np.clip(display*255.0 + 0.5,0,255).astype(np.uint8)

    if alpha is not None:
        image = np.dstack([colour,alpha])
    else:
        image = colour

    cv2.imwrite(filename,image)

    #json_dict is deliberately not taken, for the same reason the blur and the illumination
    #do not take it: a multiply moves no geometry, so the lead bounding boxes, the text
    #bounding boxes and plotted_pixels all stay valid as they are.
    return filename
