
# For ImageScope .xml annotations

import glob
import json
import math
import os
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
from scipy.ndimage.morphology import binary_erosion

from misc.utils import *

###
def parse_xml_aperio(filename):   
    """
        Aperio Imagescope
    """
    xml = ET.parse(filename)
    types_list = {} # class -> regions -> vertices
    for type_xml in xml.findall('.//Annotation'):
        pos_regions_list = []
        neg_regions_list = []
        for region_xml in type_xml.findall('.//Region'):
            vertex_list = []
            for vertex_xml in region_xml.findall('.//Vertex'):
                pos_x = float(vertex_xml.attrib['X'])
                pos_y = float(vertex_xml.attrib['Y'])
                vertex_list.append([pos_x, pos_y])
            vertex_list = np.array(vertex_list) + 0.5
            vertex_list = vertex_list.astype('int32')
            if region_xml.attrib['NegativeROA'] == '0':
                pos_regions_list.append(vertex_list)
            else:
                neg_regions_list.append(vertex_list)            
        types_list[type_xml.attrib['Name']] = [pos_regions_list, neg_regions_list]
    return types_list
###
def draw_ann_aperio(type_coords_dict, type_color_dict, canvas_size):
    """
        Aperio Imagescope
        Flatten annotations onto 1 canvas
    """
    canvas = np.zeros(canvas_size + [3,], np.uint8) 
    for typename in type_color_dict.keys():
        if typename in type_coords_dict: 
            region_color = type_color_dict[typename]     
            pos_regions_list = type_coords_dict[typename][0]  
            cv2.fillPoly(canvas, pos_regions_list, color=region_color)
            neg_regions_list = type_coords_dict[typename][1]  
            cv2.fillPoly(canvas, neg_regions_list, color=(0, 0, 0))       
    return canvas
###
svs_dir  = "/media/vqdang/Data/Workspace/KBSMC/COLON/"
anns_dir = "/media/vqdang/Data/Workspace/KBSMC/COLON/"
save_dir = '../../train/COLON_MANUAL_PATCHES/v1/'
type_color_dict = {'benign':(1, 1, 1), 
            'WD':(2, 2, 2), 'MD':(3, 3, 3), 'PD':(4, 4, 4)}

# svs_dir  = "/media/vqdang/Data/Workspace/KBSMC/PROSTATE/"
# anns_dir = "/media/vqdang/Data/Workspace/KBSMC/PROSTATE/"
# save_dir = '../../train/PROSTATE_MANUAL_PATCHES/v1/'
# type_color_dict = {'normal':(1, 1, 1), 
#             '3':(2, 2, 2), '4':(3, 3, 3), '5':(4, 4, 4)}

file_list = glob.glob(anns_dir + '*_x.txt')
file_list.sort() # ensure same order [1]

for filename in file_list: # png for base
    filename = os.path.basename(filename)
    basename = filename.split('.')[0]

    print(basename)
    with open(anns_dir + basename + '.txt') as json_file:
        rois_info = json.load(json_file)

    basename = basename.replace('_x', '')
    if basename not in ['1010713', '1010714', '1010715']:
        continue
    svs = openslide.OpenSlide(svs_dir + basename + '.svs') 

    out_dir = '%s/%s/' %(save_dir, basename)
    rm_n_mkdir(out_dir)

    svs_shape = [svs.dimensions[1], svs.dimensions[0]] 
    coords_dict = parse_xml_aperio(anns_dir + basename + '.xml')
    svs_ann = draw_ann_aperio(coords_dict, type_color_dict, svs_shape)[...,0]
    # annotations = cv2.resize(annotations, (0, 0), fx=0.25, fy=0.25)

    rois_info = rois_info['']
    for roi_idx, vertex_list in enumerate(rois_info.values()):
        for vertex_idx, vertex in enumerate(vertex_list):
            vertex_list[vertex_idx] = [float(vertex['X']), 
                                       float(vertex['Y'])]
        vertex_list = np.array(vertex_list) + 0.5
        vertex_list = vertex_list.astype('int32')

        min_x = np.amin(vertex_list[...,0])
        max_x = np.amax(vertex_list[...,0])
        min_y = np.amin(vertex_list[...,1])
        max_y = np.amax(vertex_list[...,1])

        top_corner = [min_x, min_y]
        w, h = box_size = [max_x-min_x, max_y-min_y]

        # TODO: need a way to code the core location into the patches to back track
        roi_region = svs.read_region(top_corner, 0, box_size)
        roi_region = np.array(roi_region, dtype=np.uint8)[...,:3]
        roi_ann_region = svs_ann[min_y:max_y, min_x:max_x]

        # plt.subplot(1,2,1)
        # plt.imshow(roi_region)
        # plt.subplot(1,2,2)
        # plt.imshow(roi_ann_region)
        # plt.show()

        # set the majority pixel-wise label to be the whole patch label
        max_label = None
        max_label_area = 0
        for label in range(1, 5):
            label_area = (roi_ann_region == label).sum()
            if label_area > max_label_area:
                max_label_area = label_area
                max_label = label
        # assert max_label is not None, 'ROI has no annotation' 
        max_label = 1 if max_label is None else max_label # NOTE: for prostate
        roi_region = cv2.cvtColor(roi_region, cv2.COLOR_RGB2BGR)
        cv2.imwrite('%s/%03d_%d.jpg' % (out_dir, roi_idx, max_label-1), roi_region)