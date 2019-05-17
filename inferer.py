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
import importlib
from config import Config
from scipy.ndimage.morphology import binary_erosion
from misc.utils import *

class Inferer(Config):
    def infer_step(self, net, batch):
        net.eval() # infer mode

        imgs = torch.FloatTensor(batch) # batch is NHWC
        imgs = imgs.permute(0, 3, 1, 2) # to NCHW

        # push data to GPUs and convert to float32
        imgs = imgs.to('cuda').float()

        # -----------------------------------------------------------
        with torch.no_grad(): # dont compute gradient
            logit = net(imgs)[1] # forward
            prob = nn.functional.softmax(logit, dim=1)
            prob = prob.permute(0, 2, 3, 1) # to NHWC
            return prob.cpu().numpy()

    def run(self):
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

        inf_model_path = '/media/vqdang/Data_2/dang/output/NUCLEI-ENHANCE/colon_manual/v1.0.0.1/model_net_39.pth'
        saved_state = torch.load(inf_model_path)
        pretrained_dict = saved_state.module.state_dict() # due to torch.nn.DataParallel        
        net.load_state_dict(pretrained_dict, strict=False)
        net = net.to('cuda')

        svs_code = '1010715'
        inf_imgs_dir = '/media/vqdang/Data/Workspace/KBSMC/COLON/%s/imgs/' % svs_code
        file_list = glob.glob('%s/*.jpg' % inf_imgs_dir)
        file_list.sort() # ensure same order

        inf_output_dir = '/media/vqdang/Data_2/dang/infer/nuclei_enhance/colon_manual/%s/' % svs_code
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

            img = cv2.resize(img, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
            orig_shape = img.shape[:2]
            input_img = center_pad_to(img, 5120, 5120, 255)

            # input_patches = np.reshape(input_img, (1024, 5, 1024, 5, 3))
            # input_patches = np.transpose(input_patches, (1, 3, 0, 2, 4))
            # input_patches = np.reshape(input_patches, (25, 1024, 1024, 3))
            # input_patches = list(input_patches)

            # output_prob = []
            # while len(input_patches) > 0:
            #     batches_input = input_patches[:infer_batch_size]
            #     input_patches = input_patches[infer_batch_size:]
            #     # NOTE: multiple output will fail
            #     batches_input = np.array(batches_input)
            #     batches_output = self.infer_step(net, batches_input)
            #     output_prob.extend(list(batches_output))
            # input_prob = np.array(output_prob)

            # output_prob = np.reshape(output_prob, (5, 5, 32, 32, 4))
            # output_prob = np.transpose(output_prob, (2, 0, 3, 1, 4))
            # output_prob = np.reshape(output_prob, (160, 160, 4))
            # output_prob = cv2.resize(output_prob, (5120, 5120), interpolation=cv2.INTER_CUBIC)
            # output_prob = cropping_center(output_prob, orig_shape)

            #     batches_output = self.infer_step(net, batches_input)

            output_prob = self.infer_step(net, [input_img])[0]
            output_prob = cv2.resize(output_prob, (5120, 5120), interpolation=cv2.INTER_CUBIC)
            output_prob = cropping_center(output_prob, orig_shape)
            output_pred = np.argmax(output_prob, axis=-1) / 4

            output_prob = np.transpose(output_prob, (0, 2, 1))
            output_prob = np.reshape(output_prob, (orig_shape[0], orig_shape[1] * 4))
            output_prob = (cmap(output_prob)[...,:3] * 255).astype('uint8')
            output_pred = (cmap(output_pred)[...,:3] * 255).astype('uint8')
            
            # plt.imshow(output_prob)
            # plt.show()
            # exit()

            path = path.replace('jpg', 'png')
            path = path.replace('imgs', 'anns')
            ann = cv2.imread(path)        
            ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            ann = cv2.resize(ann, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            out = np.concatenate([img, ann, output_pred, output_prob], axis=1)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            out = cv2.resize(out, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

            cv2.imwrite('%s/%s.jpg' % (inf_output_dir, basename), out)
            print('FINISH')
####

####
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    args = parser.parse_args()

    inferer = Inferer()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu       
    nr_gpus = len(args.gpu.split(','))
    inferer.run()