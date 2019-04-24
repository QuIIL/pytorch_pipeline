
import shutil
import os
import csv
import glob
import cv2
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

#####
def rm_n_mkdir(dir):
    if (os.path.isdir(dir)):
        shutil.rmtree(dir)
    os.makedirs(dir)

#### Training data
# data_files = '/mnt/dang/data/SMHTMAs/core_grade.txt'
# tma_list = ['160003', '161228', '162350', '163542', '164807']

# with open(data_files, newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=' ')
#     data = [line for line in reader]

# for tma_code in tma_list:
#     grade_list = [pair[1] for pair in data if tma_code in pair[0]]
#     grade_count =  Counter(grade_list)
#     grade_count = sorted(grade_count.items(), key=lambda pair: pair[0])
#     print(tma_code, grade_count)

# data_files = '../../cancer_detection/misc/core_colon_grade.txt'
# with open(data_files, newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=' ')
#     data = [line for line in reader]

# save_path = '/media/vqdang/Data_2/dang/train/COLON/TMA-II/CORE/'

# # rm_n_mkdir(save_path)
# for pair in data: # png for base
#     tma_name = pair[0].split('/')[-3]
#     corename = pair[0].split('/')[-1]

#     x40 = cv2.imread(pair[0])
#     x05 = cv2.resize(x40, (0,0), fx=1/8 , fy=1/8 , interpolation=cv2.INTER_CUBIC) 
#     print(pair[0], x05.shape)

#     # cv2.imwrite('%s%s_%s' % (save_path, tma_name, corename), x05)

# tma_dict = {
# "1010711" : {
#     "1": "C04 C06 C07 C09 D01 D03 D08 E08 F04 D09",
#     "2": "A02 A03 A04 A05 B01 B03 B05 B07 B10 D05 D06 D07 E01 E02 E04 E05 E09 E10 F07 F08 F09 F10",
#     "3": "A06 A07 A09 B02 B04 B09 C01 C05 C08 F05",},
# "1010712" : {
#     "1": "A08 A10 B01 B06 B07 C01 C09 D08 D09 E01 E07 E08",
#     "2": "A01 A02 A03 A04 A05 A06 A09 B05 B09 C03 C05 C06 C07 C08 D03 D04 D07 E04 E05 E06 F02 F03 F08 F10",
#     "3": "B10 D02 E10 F04 E02 F07",},
# "1010713" : {
#     "B": "A07 E03 E05 E07",
#     "1": "A06 B05 C09",
#     "2": "A03 A05 A08 A10 A09 B02 B03 B09 B10 C03 C04 C05 C06 C10 D03 D02 D10 E01 E02 E04 E06 E09 E10 F02 F03 F04 F05 F06 F07 F08 F10",
#     "3": "A01 A04 B08 C07 D08 E08",},
# "1010714": {
#     "B": "B02",
#     "1": "B01 B04 C05 E04 B04",
#     "2": "A01 A03 A04 A05 A06 A08 A10 B03 B05 B07 C02 C03 C07 D04 D05 D06 D09 D10 E02 E03 E08 E10 F05 F06",
#     "3": "B08 D08 D03 F07",},
# "1010715": {
#     "1": "A10 C08 D07 E04 E08 E09 F07",
#     "2": "A02 A03 A05 A06 A07 A09 B01 B04 B06 B08 B10 C02 C03 C04 C05 C06 C10 D06 D08 D09 D10 E05 E07 F02",
#     "3": "C07 E01 E02 F08 E10",},
# "1010716": {
#     "B": "A06 A07 B07 C07 D07 D08 E07 E08 F07 F08",
#     "1": "C05 C06 C08 F04 E02 B05",
#     "2": "B01 B02 B06 C01 D01 D03 D04 E01 E06",
#     "3": "D05 D06 E03 E04 F03",},
# }

# data_path = '/media/vqdang/Data/Workspace/KBSMC/COLON/'
# save_path = '/media/vqdang/Data_2/dang/train/COLON/TMA-I/CORE/'

# # rm_n_mkdir(save_path)
# for tma_name in tma_dict.keys():
#     core_dict = tma_dict[tma_name]
#     for label_name in core_dict.keys():
#         core_list = core_dict[label_name]
#         core_list = core_list.split(" ")
#         for core_name in core_list:
#             # x40 = cv2.imread('%s/%s/imgs/%s.jpg' % (data_path, tma_name, core_name))
#             # x05 = cv2.resize(x40, (0,0), fx=1/8 , fy=1/8 , interpolation=cv2.INTER_CUBIC) 

#             out_path = '%s/%s_%s.jpg' % (save_path, tma_name, core_name)
#             print(out_path + '\t' + label_name)
#             # cv2.imwrite(out_path, x05)