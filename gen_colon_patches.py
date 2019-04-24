
import numpy as np
import cv2
import math
import os
import glob
from scipy import io as sio
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_erosion

from misc import rm_n_mkdir, color_mask, cropping_center

####
def get_patches(x, win_size, step_size):
    msk_size = step_size

    def get_last_steps(length, msk_size, step_size):
        nr_step = math.ceil((length - msk_size) / step_size)
        last_step = (nr_step + 1) * step_size
        return int(last_step), int(nr_step + 1)
    
    im_h = x.shape[0] 
    im_w = x.shape[1]

    last_h, nr_step_h = get_last_steps(im_h, msk_size[0], step_size[0])
    last_w, nr_step_w = get_last_steps(im_w, msk_size[1], step_size[1])

    diff_h = last_h + win_size[0] - im_h
    padt = diff_h // 2
    padb = diff_h - padt

    diff_w = last_w + win_size[1] - im_w
    padl = diff_w // 2
    padr = diff_w - padl

    x = np.lib.pad(x, ((padt, padb), (padl, padr), (0, 0)), 'edge')

    #### TODO: optimize this
    sub_patches = []
    # generating subpatches from orginal
    for row in range(0, last_h, step_size[0]):
        for col in range (0, last_w, step_size[1]):
            win = x[row:row+win_size[0], 
                    col:col+win_size[1]]
            assert win.shape[0] == win_size[0] and \
                   win.shape[1] == win_size[1], "Extracted Patch's Size Incorrect"
            sub_patches.append(win)
    return sub_patches

####

step_size = [2048, 2048]
win_size  = [2048, 2048]

data_dir = '/media/vqdang/Data/Workspace/KBSMC/COLON/'
save_dir = '/media/vqdang/Data_2/train/COLON_MICCAI2019/2048x2048_tma1_rgb2/'
tma_list = ['1010711', '1010712', '1010713', '1010714', '1010715', '1010716']

for tma_code in tma_list:   
    tma_save_dir = '%s/%s/' % (save_dir, tma_code)
    imgs_list = glob.glob('%s/%s/imgs/*.jpg' % (data_dir, tma_code))
    imgs_list.sort()

    rm_n_mkdir(tma_save_dir)
    for img_path in imgs_list[1:]:
        print (img_path, ' ------ ', end='', flush=True)
        filename = os.path.basename(img_path)
        basename = filename.split('.')[0]
        
        img = cv2.imread(img_path)

        # nuc_path = '%s/%s/nuc+xy/%s.mat' % (data_dir, tma_code, basename)
        # nuc = np.squeeze(sio.loadmat(nuc_path)['result'])

        # # NOTE: temporary fix the range for nuclei map from nuc+xy
        # nuc_ch = np.array(nuc[...,0], dtype=np.float32)
        # nuc_ch = (nuc_ch - np.min(nuc_ch))/ (np.max(nuc_ch) - np.min(nuc_ch))
        # nuc_ch = (nuc_ch * 255.0).astype('uint8')
        # nuc[...,0] = nuc_ch

        # nuc_path = '%s/%s/nucs/%s.npy' % (data_dir, tma_code, basename)
        # nuc = np.expand_dims(np.squeeze(np.load(nuc_path)), axis=-1)
        # nuc = (nuc * 255.0).astype('uint8') # quantize it

        ann_path = '%s/%s/anns/%s.png' % (data_dir, tma_code, basename)
        ann = cv2.imread(ann_path)
        ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

        msk = np.zeros(ann.shape[:2] + (1,))
        msk[color_mask(ann, 125, 125, 125)] = 1
        msk[color_mask(ann,   0, 255,   0)] = 2
        msk[color_mask(ann, 255, 255,   0)] = 3
        msk[color_mask(ann, 255,   0,   0)] = 4

        # img = np.concatenate([img, nuc], axis=-1)
        msk_patches = get_patches(msk, win_size, step_size)
        img_patches = get_patches(img, win_size, step_size)

        area = np.prod(step_size)
        for idx, msk_patch in enumerate(msk_patches):
            img_patch = img_patches[idx]

            msk_central_patch = cropping_center(msk_patch, step_size)
            img_central_patch = cropping_center(img_patch, step_size)

            label = -1
            label = 0 if (msk_central_patch == 1).sum() / area >= 0.7 else label 
            label = 1 if (msk_central_patch == 2).sum() / area >= 0.7 else label 
            label = 2 if (msk_central_patch == 3).sum() / area >= 0.7 else label 
            label = 3 if (msk_central_patch == 4).sum() / area >= 0.7 else label 

            patch_gray = cv2.cvtColor(img_central_patch, cv2.COLOR_RGB2GRAY)
            thval, thmap = cv2.threshold(patch_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)  
            thmap = binary_erosion(thmap, iterations=2)

            if thmap.sum() / area <= 0.2 and label >= 0:
                cv2.imwrite('%s/%s_%03d_%d.jpg' % (tma_save_dir, basename, idx, label), img_patch)
                # np.save('%s/%s_%03d_%d.npy' % (tma_save_dir, basename, idx, label), img_patch)
        print ('FINISH')
