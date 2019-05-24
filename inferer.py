import os
import glob
import shutil
import argparse

import cv2
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import misc
import dataset
import torch.utils.data as data

import importlib
from config import Config
from scipy.ndimage.morphology import binary_erosion
from misc.utils import *

class Inferer(Config):
    def infer_step(self, net, batch):
        net.eval() # infer mode

        imgs = batch # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2) # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        # -----------------------------------------------------------
        with torch.no_grad(): # dont compute gradient
            logit = net(imgs)[1] # forward
            prob = nn.functional.softmax(logit, dim=1)
            prob = prob.permute(0, 2, 3, 1) # to NHWC
            return prob.cpu().numpy()

    def run_tma_core(self):
        def center_pad_to(img, h, w, cval=255):
            shape = img.shape

            diff_h = h - shape[0]
            padt = diff_h // 2
            padb = diff_h - padt

            diff_w = w - shape[1]
            padl = diff_w // 2
            padr = diff_w - padl

            img = np.lib.pad(img, ((padt, padb), (padl, padr), (0, 0)), 
                            'constant', constant_values=cval)
            return img

        input_chs = 3 # TODO: dynamic config
        net_def = importlib.import_module('model.net') # dynamic import
        net = net_def.DenseNet(input_chs, self.nr_classes, seg_mode=True)

        # inf_model_path = '/media/vqdang/Data_2/dang/output/NUCLEI-ENHANCE/colon_manual/v1.0.0.1/model_net_39.pth'
        inf_model_path = '/media/vqdang/Data_2/dang/output/NUCLEI-ENHANCE/prostate_manual/v1.0.0.1/model_net_60.pth'
        saved_state = torch.load(inf_model_path)
        pretrained_dict = saved_state.module.state_dict() # due to torch.nn.DataParallel        
        net.load_state_dict(pretrained_dict, strict=False)
        net = net.to('cuda')

        svs_code = '11S-1_2(x400)'
        inf_imgs_dir = '/media/vqdang/Data/Workspace/KBSMC/PROSTATE/%s/imgs/' % svs_code
        file_list = glob.glob('%s/*.jpg' % inf_imgs_dir)
        file_list.sort() # ensure same order

        # inf_output_dir = '/media/vqdang/Data_2/dang/infer/nuclei_enhance/colon_manual/%s/' % svs_code
        inf_output_dir = '/media/vqdang/Data_2/dang/infer/nuclei_enhance/prostate_manual/%s/' % svs_code
        if not os.path.isdir(inf_output_dir):
            os.makedirs(inf_output_dir) 
        infer_batch_size = 4

        cmap = plt.get_cmap('jet')
        for filename in file_list:
            filename = os.path.basename(filename)
            basename = filename.split('.')[0]

            print(filename, ' ---- ', end='', flush=True)

            path = inf_imgs_dir + filename
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # img = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            orig_shape = img.shape[:2]
            input_img = center_pad_to(img, 3072, 3072, 255) # 5120x5120 colon, 3072x3072 prostate

            output_prob = self.infer_step(net, [input_img])[0]
            output_prob = cv2.resize(output_prob, (3072, 3072), interpolation=cv2.INTER_CUBIC)
            output_prob = cropping_center(output_prob, orig_shape)
            output_pred = np.argmax(output_prob, axis=-1) / 3

            output_prob = np.transpose(output_prob, (0, 2, 1))
            output_prob = np.reshape(output_prob, (orig_shape[0], orig_shape[1] * 3))
            output_prob = (cmap(output_prob)[...,:3] * 255).astype('uint8')
            output_pred = (cmap(output_pred)[...,:3] * 255).astype('uint8')
            
            path = path.replace('jpg', 'png')
            path = path.replace('imgs', 'anns')
            ann = cv2.imread(path)        
            ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            # ann = cv2.resize(ann, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            out = np.concatenate([img, ann, output_pred, output_prob], axis=1)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            # out = cv2.resize(out, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            cv2.imwrite('%s/%s.jpg' % (inf_output_dir, basename), out)
            print('FINISH')

    def run_wsi(self):
        import openslide
        from progress.bar import Bar as ProgressBar

        input_chs = 3 # TODO: dynamic config
        net_def = importlib.import_module('model.net') # dynamic import
        net = net_def.DenseNet(input_chs, self.nr_classes, seg_mode=True)

        inf_model_path = '/media/vqdang/Data_2/dang/output/NUCLEI-ENHANCE/colon_manual/v1.0.0.2/model_net_36.pth'
        pretrained_dict = torch.load(inf_model_path)
        pretrained_dict = pretrained_dict.module.state_dict() # due to torch.nn.DataParallel        
        net.load_state_dict(pretrained_dict, strict=False)
        net = torch.nn.DataParallel(net).to('cuda')

        inf_svs_dir = '/media/vqdang/Data/Workspace/KBSMC/COLON_WSI/'
        file_list = glob.glob('%s/*.ndpi' % inf_svs_dir)
        file_list.sort() # ensure same order

        # inf_output_dir = '/media/vqdang/Data_2/dang/infer/nuclei_enhance/colon_manual/%s/' % svs_code
        inf_output_dir = '/media/vqdang/Data_3/dang/infer/nuclei_enhance/v1.0.0.2/colon_manual/wsi/'
        if not os.path.isdir(inf_output_dir):
            os.makedirs(inf_output_dir) 

        zoom_level = 2
        cmap = plt.get_cmap('jet')
        for filepath in file_list[:20]:
            filename = os.path.basename(filepath)
            basename = filename.split('.')[0]
            print(filepath)

            svs = openslide.OpenSlide(filepath) 
            svs_shape = svs.level_dimensions[zoom_level] # width, height
            orig_svs_shape = [svs_shape[1], svs_shape[0]]
            print(svs.level_dimensions[zoom_level], svs.level_downsamples[zoom_level])

            cache_wsi_patches_dir = '/media/vqdang/Data_3/dang/infer/nuclei_enhance/colon_wsi/%s/' % basename
            if not os.path.isdir(cache_wsi_patches_dir):
                os.makedirs(cache_wsi_patches_dir) 

                for ridx in range(0, orig_svs_shape[0], 1024):
                    for cidx in range(0, orig_svs_shape[1], 1024):
                        # must use coordinate at zoom level 0
                        top_corner = (int(cidx * (2 ** zoom_level)), 
                                      int(ridx * (2 ** zoom_level))) # box size at extraction level
                        roi_region = svs.read_region(top_corner, zoom_level, (1024, 1024))
                        roi_region = np.array(roi_region, dtype=np.uint8)[...,:3]
                        roi_region = cv2.cvtColor(roi_region, cv2.COLOR_RGB2BGR)
                        cv2.imwrite('%s/%d_%d.jpg' % (cache_wsi_patches_dir, ridx, cidx), roi_region)

            patch_list = glob.glob('%s/*.jpg' % cache_wsi_patches_dir)
            infer_dataset = dataset.DatasetSerialWSI(patch_list)

            dataloader = data.DataLoader(infer_dataset, 
                            num_workers=2, 
                            batch_size=16, 
                            drop_last=False)

            svs_shape = [int(int(orig_svs_shape[0] / 1024 + 1.0) * 1024 / 8), 
                         int(int(orig_svs_shape[1] / 1024 + 1.0) * 1024 / 8)]
            out_svs = np.zeros(svs_shape, dtype=np.float32)
            pbar = ProgressBar('Processing', max=len(dataloader), width=48)
            for batch_data in dataloader:
                imgs_input, imgs_path = batch_data  
                output_prob = self.infer_step(net, imgs_input)
                for idx, patch_path in enumerate(imgs_path):
                    patch_loc = os.path.basename(patch_path).split('.')[0].split('_')
                    patch_loc = [int(int(patch_loc[0]) / 8), int(int(patch_loc[1]) / 8)]
                    if output_prob[idx].shape[0] < 128 or \
                        output_prob[idx].shape[1] < 128: continue
                    # output_patch = output_prob[idx]  
                    output_patch = np.argmax(output_prob[idx], axis=-1)              
                    out_svs[patch_loc[0] : patch_loc[0] + 128,
                            patch_loc[1] : patch_loc[1] + 128] = output_patch
                pbar.next()
            pbar.finish()
            out_svs = out_svs[:orig_svs_shape[0], :orig_svs_shape[1]]                
            out_svs = (cmap((out_svs + 1) / 4)[...,:3] * 255).astype('uint8')
            # svs_shape = out_svs.shape
            # out_svs = np.transpose(out_svs, (0, 2, 1))
            # out_svs = np.reshape(out_svs, (svs_shape[0], svs_shape[1] * 4))
            # out_svs = (cmap(out_svs)[...,:3] * 255).astype('uint8')
            out_svs = cv2.resize(out_svs, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)   
            out_svs = cv2.cvtColor(out_svs, cv2.COLOR_RGB2BGR)
            cv2.imwrite('%s/%s.jpg' % (inf_output_dir, basename), out_svs)
####

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()

    inferer = Inferer()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu       
    nr_gpus = len(args.gpu.split(','))
    inferer.run_wsi()