import os
import zlib
import cv2
import numpy as np
from math import ceil
from linear_light import srgb_to_linear, linear_to_srgb, mean_preserving_gain

#Octaves summed into the height field, and the amplitude ratio between one octave and
#the next. Three octaves are enough for paper: a dominant fold scale, a secondary one
#and a fine waviness.
CRUMPLE_OCTAVES = 3
CRUMPLE_OCTAVE_PERSISTENCE = 0.5

#The noise grid of every octave is generated with a margin of whole cells around the
#page and cropped afterwards, and never with fewer cells than the minimum. At
#crumple_scale_cm 15 the natural grid is about 3 cells across an 11 inch page, which
#leaves cubic upsampling without support: the field degenerates into a plane and reads
#as a tilted sheet rather than as crumpled paper.
CRUMPLE_GRID_MARGIN_CELLS = 2
CRUMPLE_MIN_GRID_CELLS = 4

#Surface slope (rise over run) at crumple_amplitude 1.0, as an RMS over the page, and
#in plane displacement at the same amplitude, in mm of paper. The amplitude scales the
#SLOPE rather than the height of the field: with a fixed height the slope would go as
#1/crumple_scale_cm, and amplitude and scale would then both drive the shading
#contrast and compete for the same target metric.
CRUMPLE_MAX_SLOPE = 0.23
CRUMPLE_MAX_DISPLACEMENT_MM = 0.6

#Width of the border over which the displacement fades out, as a multiple of the largest
#displacement in the field.
CRUMPLE_TAPER_MARGIN_FACTOR = 2.0

#Fixed point iterations used to carry an annotated point through the deformation.
CRUMPLE_INVERSE_ITERATIONS = 3

#Resampling kernel of the deformation. The trace is about 1.8 px wide at 300 dpi, right
#at the sampling limit, so a sub pixel warp costs sharpness whatever the kernel; measured
#on this render, a bare half pixel shift with no shading at all takes glcm_contrast from
#75.3 to 42.7 with INTER_LINEAR and to 50.8 with INTER_LANCZOS4. The loss is a fixed cost
#of entering the warp and barely depends on the displacement magnitude.
CRUMPLE_INTERPOLATION = cv2.INTER_LANCZOS4

#Elevation of the light above the plane of the sheet. The azimuth is a parameter,
#since the illumination gradient has to reuse it.
CRUMPLE_LIGHT_ELEVATION_DEG = 40.0

#Wrap term on the diffuse response: the surface is lit by (N.L + wrap)/(1 + wrap) rather
#than by N.L alone. A sheet of paper on a desk is not lit by a point source in a void; it
#also receives bounce light from the room, so a facet turned away from the window darkens
#but keeps its structure. Without the wrap every facet with N.L below zero pins to the
#same floor, and above crumple_amplitude 2 the deep shadows collapse into flat blobs with
#hard edges that read as stains rather than as paper.
#It costs contrast per unit of amplitude - the relative swing goes as 1/(sin(elevation) +
#wrap) - which is why CRUMPLE_MAX_SLOPE was raised from 0.20 to keep the calibration of
#amplitude 1.0 where it was.
CRUMPLE_LIGHT_WRAP = 0.5
#Hard floor kept only as a safety net for the extreme tail; with the wrap in place it
#almost never binds.
CRUMPLE_MIN_LAMBERT = 0.02

def noise_octave(shape,cell_px,rng):
    #One octave of smooth noise: white noise on a coarse grid of cell_px spacing, raised
    #to full resolution by cubic interpolation. Cheaper than blurring at full resolution
    #with a sigma of hundreds of pixels, and smooth by construction.
    height,width = shape
    cells_y = max(int(ceil(height/cell_px)),CRUMPLE_MIN_GRID_CELLS)
    cells_x = max(int(ceil(width/cell_px)),CRUMPLE_MIN_GRID_CELLS)
    grid_y = cells_y + 2*CRUMPLE_GRID_MARGIN_CELLS
    grid_x = cells_x + 2*CRUMPLE_GRID_MARGIN_CELLS
    #The canvas is sized so that the inner cells_y by cells_x cells span the page
    #exactly; the margin cells fall outside and are cropped away, which keeps the cubic
    #kernel supported at the borders and away from edge extrapolation.
    canvas_y = int(round(height*grid_y/cells_y))
    canvas_x = int(round(width*grid_x/cells_x))
    noise = rng.standard_normal((grid_y,grid_x)).astype(np.float32)
    canvas = cv2.resize(noise,(canvas_x,canvas_y),interpolation=cv2.INTER_CUBIC)
    offset_y = (canvas_y - height)//2
    offset_x = (canvas_x - width)//2
    return canvas[offset_y:offset_y + height,offset_x:offset_x + width]

def height_field(shape,scale_px,rng):
    #Height of the paper surface, normalised to unit standard deviation. The dominant
    #octave sits at scale_px; each further octave halves the feature size and its weight.
    field = np.zeros(shape,dtype=np.float32)
    weight = 1.0
    total = 0.0
    for octave in range(CRUMPLE_OCTAVES):
        cell_px = max(scale_px/(2.0**octave),2.0)
        field += weight*noise_octave(shape,cell_px,rng)
        total += weight
        weight *= CRUMPLE_OCTAVE_PERSISTENCE
    field /= total
    deviation = float(field.std())
    if deviation <= 0:
        return None
    return field/deviation

def edge_taper(shape,margin_px):
    #Fade the displacement to zero at the border of the sheet. A large displacement asks
    #cv2.remap for samples from outside the image, and the border pixel it replicates is
    #the black axes frame matplotlib draws around the render: it smears into a dark band
    #down the edge, which is plainly visible above crumple_amplitude 2. Holding the
    #outline still while the interior deforms is also the honest reading of a sheet lying
    #flat on a desk.
    height,width = shape
    def ramp(size):
        distance = np.minimum(np.arange(size),np.arange(size)[::-1])/max(margin_px,1.0)
        distance = np.clip(distance,0.0,1.0)
        #smoothstep, so the taper has no visible seam where it reaches full strength
        return (distance*distance*(3.0 - 2.0*distance)).astype(np.float32)
    return ramp(height)[:,np.newaxis]*ramp(width)[np.newaxis,:]

def sample_field(field,rows,cols):
    #Bilinear read of a per pixel field at arbitrary sub pixel positions, used to carry
    #the annotations along with the paper.
    height,width = field.shape
    rows = np.clip(rows,0,height - 1)
    cols = np.clip(cols,0,width - 1)
    row0 = np.floor(rows).astype(int)
    col0 = np.floor(cols).astype(int)
    row1 = np.minimum(row0 + 1,height - 1)
    col1 = np.minimum(col0 + 1,width - 1)
    dr = rows - row0
    dc = cols - col0
    return (field[row0,col0]*(1 - dr)*(1 - dc) + field[row0,col1]*(1 - dr)*dc
            + field[row1,col0]*dr*(1 - dc) + field[row1,col1]*dr*dc)

def invert_displacement(rows,cols,displacement_x,displacement_y):
    #Where a feature that sat at (rows,cols) ends up after the deformation.
    #cv2.remap takes a BACKWARD map - the warped image at p is read from p + D(p) - so the
    #new position p is the solution of p + D(p) = q. The single step p = q - D(q) is only
    #the first order answer and drifts wherever D changes over the distance the point
    #moves: measured at crumple_amplitude 3 it left 17% of the annotated samples off the
    #trace. Three fixed point iterations close it.
    new_rows = rows
    new_cols = cols
    for _ in range(CRUMPLE_INVERSE_ITERATIONS):
        shift_rows = sample_field(displacement_y,new_rows,new_cols)
        shift_cols = sample_field(displacement_x,new_rows,new_cols)
        new_rows = rows - shift_rows
        new_cols = cols - shift_cols
    return new_rows,new_cols

def displace_annotations(json_dict,displacement_x,displacement_y,pad_x,pad_y):
    #Move the stored geometry with the paper.
    #The annotations are stored in the coordinate frame of the unpadded render, while the
    #displacement field covers the padded image, so the padding offset is added when the
    #field is read and removed again when the result is written back. Rewriting
    #json_dict['leads'] through convert_bounding_boxes_to_dict is not an option: that
    #helper emits integer keys while read_leads, which get_augment calls next, indexes
    #the corners with string keys.
    for lead in json_dict['leads']:
        for key in ('lead_bounding_box','text_bounding_box'):
            if key not in lead:
                continue
            corners = lead[key]
            names = list(corners)
            points = np.array([corners[name] for name in names],dtype=float)
            rows,cols = invert_displacement(points[:,1] + pad_y,points[:,0] + pad_x,
                                            displacement_x,displacement_y)
            for index,name in enumerate(names):
                corners[name] = [round(float(cols[index]) - pad_x,2),
                                 round(float(rows[index]) - pad_y,2)]
        pixels = np.asarray(lead['plotted_pixels'],dtype=float)
        if pixels.size == 0:
            continue
        rows,cols = invert_displacement(pixels[:,1] + pad_y,pixels[:,0] + pad_x,
                                        displacement_x,displacement_y)
        lead['plotted_pixels'] = [[round(float(c) - pad_x,2),round(float(r) - pad_y,2)]
                                  for r,c in zip(rows,cols)]

    if 'gridpoints' in json_dict:
        #Page-level, top-level annotation (not nested under 'leads'), same [x,y] /
        #unpadded-render-frame convention as plotted_pixels above, so the identical
        #pad-then-invert-then-unpad round trip applies unchanged. gridpoints_mask and
        #gridpoints_reference_hw are untouched here: the mask is finalised once, after
        #the whole pipeline, and the reference size is a page-level scalar pair that no
        #per-point warp applies to.
        points = np.asarray(json_dict['gridpoints'],dtype=float)
        if points.size > 0:
            rows,cols = invert_displacement(points[:,1] + pad_y,points[:,0] + pad_x,
                                            displacement_x,displacement_y)
            json_dict['gridpoints'] = [[round(float(c) - pad_x,2),round(float(r) - pad_y,2)]
                                       for r,c in zip(rows,cols)]

#Main function to deform and shade the sheet of paper
def get_crumpled(input_file,resolution,crumple_amplitude,crumple_scale_cm,
                 illum_azimuth_deg,seed=-1,start_index=-1,json_dict=None):
    #Crumple the paper: a single height field drives BOTH the geometric deformation and
    #the shading. Generating the two independently is what makes creases fail to line up
    #with their own shadows, and the result then reads as an overlaid texture rather than
    #as a deformed sheet.
    #crumple_amplitude 0 is a no op: the file is left exactly as it was found.
    filename = input_file
    if crumple_amplitude <= 0 or crumple_scale_cm <= 0:
        return filename

    image = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
    if image is None:
        return filename
    if image.ndim == 2:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
    height,width = image.shape[:2]
    alpha = image[:,:,3] if image.shape[2] == 4 else None
    colour = image[:,:,:3].astype(np.float32)/255.0

    #Both the feature scale and the displacement are paper lengths, so the render is
    #invariant to the output resolution, as the trace thickness already is.
    px_per_mm = resolution/25.4
    #A stream of its own, independent of the global random module, so that turning the
    #crumple on does not shift the grid colours, the font choice or the colour
    #temperature drawn later in get_augment. The record name enters the seed so that the
    #sheets of a batch are not all crumpled identically.
    record_key = zlib.crc32(os.path.basename(filename).encode('utf-8'))
    rng = np.random.default_rng([abs(seed),abs(start_index),record_key])

    field = height_field((height,width),crumple_scale_cm*10.0*px_per_mm,rng)
    if field is None:
        return filename

    gradient_y,gradient_x = np.gradient(field)
    gradient_rms = float(np.sqrt(np.mean(gradient_x**2 + gradient_y**2)))
    if gradient_rms <= 0:
        return filename

    #The same raw gradient is scaled twice: once to the surface slope that sets the
    #shading, once to the in plane displacement in pixels. Both scale with the amplitude,
    #so shadows and creases can never drift apart.
    slope_scale = crumple_amplitude*CRUMPLE_MAX_SLOPE/gradient_rms
    slope_x = gradient_x*slope_scale
    slope_y = gradient_y*slope_scale
    displacement_scale = (crumple_amplitude*CRUMPLE_MAX_DISPLACEMENT_MM*px_per_mm
                          /gradient_rms)
    displacement_x = gradient_x*displacement_scale
    displacement_y = gradient_y*displacement_scale
    taper = edge_taper((height,width),
                       CRUMPLE_TAPER_MARGIN_FACTOR*float(np.hypot(displacement_x,displacement_y).max()))
    displacement_x = displacement_x*taper
    displacement_y = displacement_y*taper

    #Surface normals from the same gradient, and a lambertian N.L term. The image y axis
    #points down, so the azimuth is measured in image coordinates.
    azimuth = np.radians(illum_azimuth_deg)
    elevation = np.radians(CRUMPLE_LIGHT_ELEVATION_DEG)
    light = np.array([np.cos(azimuth)*np.cos(elevation),
                      np.sin(azimuth)*np.cos(elevation),
                      np.sin(elevation)],dtype=np.float32)
    norm = np.sqrt(slope_x**2 + slope_y**2 + 1.0)
    lambert = (-slope_x*light[0] - slope_y*light[1] + light[2])/norm
    lambert = (lambert + CRUMPLE_LIGHT_WRAP)/(1.0 + CRUMPLE_LIGHT_WRAP)
    lambert = np.maximum(lambert,CRUMPLE_MIN_LAMBERT)
    #Normalised to mean 1.0. Without this the shading would act as a global gain and
    #compete with the exposure parameter of the photometric stage.
    shading = (lambert/float(lambert.mean())).astype(np.float32)

    grid_y,grid_x = np.mgrid[0:height,0:width]
    map_x = (grid_x + displacement_x).astype(np.float32)
    map_y = (grid_y + displacement_y).astype(np.float32)

    #Resampling and shading both happen in linear light. Resampling in sRGB space
    #darkens high contrast edges, and this image is almost entirely black trace on white
    #paper, so the error would land exactly on the signal.
    linear = srgb_to_linear(colour)
    reference = linear
    linear = cv2.remap(linear,map_x,map_y,CRUMPLE_INTERPOLATION,borderMode=cv2.BORDER_REPLICATE)
    #Lanczos can overshoot at the edge of the trace; in linear light the overshoot has to
    #be clipped before the shading multiplies it, or a negative undershoot would come back
    #as a dark halo.
    np.clip(linear,0.0,1.0,out=linear)
    linear *= shading[:,:,np.newaxis]*mean_preserving_gain(linear,shading,reference)
    colour = np.clip(linear_to_srgb(linear)*255.0 + 0.5,0,255).astype(np.uint8)

    if alpha is not None:
        alpha = cv2.remap(alpha,map_x,map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        image = np.dstack([colour,alpha])
    else:
        image = colour

    cv2.imwrite(filename,image)

    if json_dict is not None and 'leads' in json_dict:
        #The render is padded after the coordinates are written, so the annotation frame
        #is offset from the image frame by half the padding.
        pad_x = (width - json_dict.get('width',width))/2.0
        pad_y = (height - json_dict.get('height',height))/2.0
        displace_annotations(json_dict,displacement_x,displacement_y,pad_x,pad_y)

    return filename

def light_azimuth(illum_azimuth_deg,input_file,seed=-1,start_index=-1):
    #Resolve the direction the scene is lit from, in degrees. A negative value means one
    #azimuth per image; anything else is used as given.
    #The draw uses a stream of its own rather than the global random module, so that a
    #randomised azimuth does not shift the grid colours or the colour temperature. The
    #illumination gradient has to call this same function: two different light directions
    #in one image is physically impossible and plainly visible.
    if illum_azimuth_deg >= 0:
        return float(illum_azimuth_deg%360.0)
    record_key = zlib.crc32(os.path.basename(input_file).encode('utf-8'))
    return float(np.random.default_rng([abs(seed),abs(start_index),record_key,1]).uniform(0,360))
