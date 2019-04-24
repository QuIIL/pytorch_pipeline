
import os
import re
import cv2
import csv
import shutil
import numpy as numpy
import scipy.io as sio

x_path = "/media/vqdang/Data/Workspace/SMHTMAs/"
y_path = "/media/vqdang/Data/Workspace/SMHTMAs/Labels/"

x_out_path = '/media/vqdang/Data_2/dang/data/SMHTMAs/x0.5/'
y_out_file = '/media/vqdang/Data_2/dang/data/SMHTMAs/core_bc.txt'

scale_fctr = 0.5
write_imgs = False

y_file_ptr = open(y_out_file,'w')
for root, dirnames, filenames in os.walk(y_path):
    for filename in filenames: # png for base
        if re.search("\.(csv)$", filename):
            basename = os.path.splitext(filename)[0]

            with open(y_path + filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                next(reader) # skip the 1st header line
                for line in reader:

                    label = 0
                    if line[3] == 'C':
                        label = int(line[4])
                    elif line[3] != 'B': # images with no labels
                        continue 

                    img_name = basename + '_Grp' + ''.join(line[:3]) 
                    src_path = "{0}/{1}/core_img/{2}.png".format(x_path, basename, img_name)
                    img_path = x_out_path + img_name + '.png' # new path
                    if not os.path.isfile(src_path):  # make sure file exist in source 
                        print(src_path, '\t[NOT FOUND | NO LABEL]')
                        continue
                    else:
                        y_file_ptr.write(img_path + ' %d\n' % label)
                        print(src_path, img_path, end='\t')

                    if write_imgs:
                        img = cv2.imread(src_path, cv2.IMREAD_COLOR)
                        img = cv2.resize(img, (0,0), fx=scale_fctr, fy=scale_fctr,
                                                interpolation=cv2.INTER_LANCZOS4) 
                        cv2.imwrite(img_path, img)
                        print(img.shape)
                    print()
y_file_ptr.close()