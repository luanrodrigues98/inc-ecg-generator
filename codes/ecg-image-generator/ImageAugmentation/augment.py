import imageio, json
from PIL import Image
import argparse
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from helper_functions import read_leads, convert_bounding_boxes_to_dict, rotate_bounding_box, get_lead_pixel_coordinate, rotate_points, crop_points, crop_transform_params
import numpy as np
import matplotlib.pyplot as plt
import os, sys, argparse
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from math import ceil 
import time
import random

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_directory', type=str, required=True)
    parser.add_argument('-i', '--input_file', type=str, required=True)
    parser.add_argument('-o', '--output_directory', type=str, required=True)
    parser.add_argument('-r','--rotate',type=int,default=25)
    parser.add_argument('-n','--noise',type=int,default=25)
    parser.add_argument('-c','--crop',type=float,default=0.01)
    parser.add_argument('-t','--temperature',type=int,default=6500)
    return parser

# Main function for running augmentations
def get_augment(input_file,output_directory,rotate=25,noise=25,crop=0.01,temperature=6500,bbox=False, store_text_bounding_box=False, json_dict=None):
    filename = input_file
    image = Image.open(filename)
    
    image = np.array(image)
    
    lead_bbs = []
    leadNames_bbs = []
    
         
    lead_bbs, leadNames_bbs, lead_bbs_labels, startTime_bbs, endTime_bbs, plotted_pixels = read_leads(json_dict['leads'])
    
    if bbox:
        lead_bbs = BoundingBoxesOnImage(lead_bbs, shape=image.shape)
    if store_text_bounding_box:
        leadNames_bbs = BoundingBoxesOnImage(leadNames_bbs, shape=image.shape)
    
    images = [image[:, :, :3]]
    h, w, _ = image.shape
    rot = random.randint(-rotate, rotate)
    crop_sample = random.uniform(0, crop)
    #Augment in a sequential manner. Create an augmentation object
    #A temperature of 0 or less REMOVES the colour temperature step instead of passing a
    #neutral kelvin to it, and the difference matters: imgaug's table has no exactly
    #neutral entry. Measured, 6500 K - the closest there is - still leaves white paper at
    #[255,249,253] and mid grey at [128,125,127], a residual of about 2%, which is the same
    #order as a wb_b of 1.02 and would go on competing with the white balance stage for
    #ownership of the colour cast. Off is the only setting that is actually neutral.
    #0 is also not a colour temperature, so it cannot collide with a value a caller means.
    steps = [iaa.Affine(rotate=rot),
             iaa.AdditiveGaussianNoise(scale=(noise, noise)),
             iaa.Crop(percent=crop_sample)]
    if temperature > 0:
        steps.append(iaa.ChangeColorTemperature(temperature))
    seq = iaa.Sequential(steps)
    
    images_aug = seq(images=images)

    if bbox:
        augmented_lead_bbs = rotate_bounding_box(lead_bbs, [w/2,h/2], -rot)
    else:
        augmented_lead_bbs = []    
    if store_text_bounding_box:
        augmented_leadName_bbs = rotate_bounding_box(leadNames_bbs, [w/2,h/2], -rot)
    else:
        augmented_leadName_bbs = []   

    rotated_pixel_coordinates = rotate_points(plotted_pixels, [w/2, h/2], -rot)

    #iaa.Crop(percent=crop_sample, keep_size=True) - the default, unchanged here - crops
    #crop_sample off every side and then resizes the remainder back up to (h,w): an
    #offset AND a scale, which nothing above this line accounts for. Previously only the
    #rotation was corrected and plotted_pixels silently drifted out of alignment with
    #the image whenever crop_sample > 0. crop_sample is a plain scalar (not a tuple or
    #StochasticParameter), so all four sides crop by the same, already-known fraction -
    #no extra imgaug RNG draw needs to be read back to reproduce it analytically.
    crop_top, crop_left, crop_scale_y, crop_scale_x = crop_transform_params(h, w, crop_sample)
    rotated_pixel_coordinates = crop_points(rotated_pixel_coordinates, crop_top, crop_left,
                                            crop_scale_y, crop_scale_x)

    if bbox or store_text_bounding_box:
        json_dict['leads'] = convert_bounding_boxes_to_dict(augmented_lead_bbs, augmented_leadName_bbs, lead_bbs_labels, startTime_bbs, endTime_bbs, rotated_pixel_coordinates)

    #--store_gridpoints ground truth. Top-level key, not nested under 'leads', so it is
    #updated unconditionally here rather than being gated behind bbox/store_text_bounding_box
    #the way the leads writeback above is.
    if json_dict.get('gridpoints'):
        gridpoints = rotate_points([json_dict['gridpoints']], [w/2, h/2], -rot)
        gridpoints = crop_points(gridpoints, crop_top, crop_left, crop_scale_y, crop_scale_x)
        json_dict['gridpoints'] = gridpoints[0].tolist()
    if json_dict.get('gridpoints_reference_hw'):
        #Rotation does not change the reference page size, only the crop-driven zoom
        #does - the same per-axis scale the point transform above uses.
        ref_h, ref_w = json_dict['gridpoints_reference_hw']
        json_dict['gridpoints_reference_hw'] = [round(ref_h*crop_scale_y, 2),
                                                round(ref_w*crop_scale_x, 2)]

    head, tail = os.path.split(filename)

    f = os.path.join(output_directory,tail)
    plt.imsave(fname=f,arr=images_aug[0])

    return f

