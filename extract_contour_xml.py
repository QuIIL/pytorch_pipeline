
# For ImageScope .xml annotations

import glob
import os
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt

import cv2
import math
import openslide
import numpy as np
from scipy.ndimage.morphology import binary_erosion

from misc.utils import *

###
def get_last_steps(length, msk_size, step_size):
    nr_step = math.ceil((length - msk_size) / step_size)
    last_step = (nr_step + 1) * step_size
    return int(last_step), int(nr_step + 1)

###
win_size  = [1024, 1024]
step_size = [ 512,  512]
###
svs_dir  = "/media/vqdang/Data/Workspace/KBSMC/PROSTATE/"
anns_dir = "/media/vqdang/Data/Workspace/KBSMC/PROSTATE/"
save_dir = '../../train/PROSTATE_ASAN/%dx%d_%dx%d' % \
                (win_size[0], win_size[1], step_size[0], step_size[1])

file_list = glob.glob(anns_dir + '*.xml')
file_list.sort() # ensure same order [1]

area = np.prod(step_size)
for filename in file_list: # png for base
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]

    print(basename)
    xml = ET.parse(anns_dir + basename + '.xml')
    svs = openslide.OpenSlide(svs_dir + basename + '.svs') 

    out_dir = '%s/%s/' %(save_dir, basename)
    rm_n_mkdir(out_dir)

    for type_idx, type_xml in enumerate(xml.findall('.//Annotation')):
        type_name = type_xml.attrib['Name']
        if type_name not in ['3', '4', '5', 'benign']:
            continue

        for region_idx, region_xml in enumerate(type_xml.findall('.//Region')):
            vertex_list = []
            for vertex_xml in region_xml.findall('.//Vertex'):
                attrib = vertex_xml.attrib
                vertex_list.append([float(attrib['X']), 
                                 float(attrib['Y'])])
            vertex_list = np.array(vertex_list) + 0.5
            vertex_list = vertex_list.astype('int32')

            min_x = np.amin(vertex_list[...,0])
            max_x = np.amax(vertex_list[...,0])
            min_y = np.amin(vertex_list[...,1])
            max_y = np.amax(vertex_list[...,1])

            top_corner = [min_x, min_y]
            w, h = box_size = [max_x-min_x, max_y-min_y]
            if w * h < area: continue

            last_h, nr_step_h = get_last_steps(h, win_size[0], step_size[0])
            last_w, nr_step_w = get_last_steps(w, win_size[1], step_size[1])

            diff_h = win_size[0] - step_size[0]
            padt = diff_h // 2
            padb = last_h + win_size[0] - h

            diff_w = win_size[1] - step_size[1]
            padl = diff_w // 2
            padr = last_w + win_size[1] - w

            # calculate the patch to extract
            extract_corner = (top_corner[0] - padl, top_corner[1] - padt)
            extract_box_size = (box_size[0] + padl + padr, 
                                box_size[1] + padt + padb)
            scan_region = svs.read_region(extract_corner, 0, extract_box_size)
            scan_region = np.array(scan_region, dtype=np.uint8)[...,:3]

            # shift origin to draw
            vertex_list[...,0] -= (min_x - padl) 
            vertex_list[...,1] -= (min_y - padt) 
            canvas = np.zeros(scan_region.shape[:2], np.uint8)
            # fill both the inner area and contour with idx+1 color
            cv2.drawContours(canvas, [vertex_list], 0, 1, -1)

            gray_region = cv2.cvtColor(scan_region, cv2.COLOR_RGB2GRAY)
            thval, thmap = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
            thmap = binary_erosion(thmap, iterations=2)
            canvas[thmap > 0] = 0 # setting up ignored region

            canvas = np.expand_dims(canvas, axis=-1)
            scan_region = cv2.cvtColor(scan_region, cv2.COLOR_RGB2BGR)
            x = np.concatenate([scan_region, canvas], axis=-1)
            #### TODO: optimize this
            sub_patches = []
            # generating subpatches from orginal
            for row in range(0, last_h, step_size[0]):
                for col in range (0, last_w, step_size[1]):
                    win = x[row:row+win_size[0], 
                            col:col+win_size[1]]
                    cen = cropping_center(win, step_size)
                    if cen[...,-1].sum() / area > 0.7:
                        sub_patches.append(win)
            #
            for patch_idx, patch in enumerate(sub_patches):
                cv2.imwrite('%s/%d_%d_%s.jpg' % (out_dir, region_idx, patch_idx, type_name), patch[...,:3])