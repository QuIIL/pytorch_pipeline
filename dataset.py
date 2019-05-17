import csv
import glob
import random
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms
from torchvision.utils import make_grid


####
class DatasetSerial(data.Dataset):
    @staticmethod
    def _isimage(image, ends):
        return any(image.endswith(end) for end in ends)
               
    def __init__(self, pair_list, shape_augs=None, input_augs=None, has_aux=False):
        self.has_aux = has_aux
        self.pair_list = pair_list
        self.shape_augs = shape_augs
        self.input_augs = input_augs

    def __getitem__(self, idx):

        pair = self.pair_list[idx]

        input_img = cv2.imread(pair[0])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_label = pair[1] # normal is 0

        # shape must be deterministic so it can be reused
        shape_augs = self.shape_augs.to_deterministic()
        input_img = shape_augs.augment_image(input_img)

        # additional augmentation just for the input
        if self.input_augs is not None:
            input_img = self.input_augs.augment_image(input_img)

        return input_img, img_label
        
    def __len__(self):
        return len(self.pair_list)
    
####
def prepare_smhtma_data(fold_idx=0):
    assert fold_idx < 5, "Currently only support 5 fold, each fold is 1 TMA"

    data_files = '/media/vqdang/Data_2/dang/train/SMHTMAs/core_grade.txt'
    tma_list = ['160003', '161228', '162350', '163542', '164807']

    # label -1 means exclude
    grade_pair = {'B': 0,
                '3+3': 1, '3+4': 1, '4+3': 2,
                '4+4': 2, '3+5': 2, '5+3': 2,
                '4+5': 2, '5+4': 2, '5+5': 2}
    with open(data_files, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        data_info = [line for line in reader] # [[path, label], etc]
    data_info = [list(t) for t in zip(*data_info)] # [path_list, label_list]
    data_info[1] = [grade_pair[label] for label in data_info[1]]
    data_info = [list(t) for t in zip(*data_info)] # [[path, label], etc]
    
    valid_tma = [tma_list[fold_idx]]
    train_tma = [tma_list[idx] for idx in range(0, 5) if idx != fold_idx]
    print(valid_tma, train_tma)

    train_pairs = []
    for tma in train_tma:
        filtered = [pair for pair in data_info if tma in pair[0]]
        train_pairs.extend(filtered)

    valid_pairs = []
    for tma in valid_tma:
        filtered = [pair for pair in data_info if tma in pair[0]]
        valid_pairs.extend(filtered)

    # filter so only cancer samples remain
    train_pairs = [pair for pair in train_pairs if pair[1] >= 0]
    valid_pairs = [pair for pair in valid_pairs if pair[1] >= 0]
    return train_pairs, valid_pairs

####
def prepare_colon_data(fold_idx=0):
    assert fold_idx < 5, "Currently only support 5 fold, each fold is 1 TMA"

    tma_list = ['1010711', '1010712', '1010713', '1010714', '1010715', '1010716']

    file_list = []
    label_list = []
    for tma_code in tma_list:
        tma_file_list = glob.glob('../../../train/COLON_MICCAI2019/2048x2048_tma_1_rgb/%s/*.jpg' % tma_code)
        tma_label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in tma_file_list]
        file_list.extend(tma_file_list)
        label_list.extend(tma_label_list)
    pairs_list = list(zip(file_list, label_list))
    # [(0, 139), (1, 235), (2, 645), (3, 194)] highly imbalance

    train_fold = []
    valid_fold = []
    skf = StratifiedKFold(n_splits=5, random_state=5, shuffle=False)
    for train_index, valid_index in skf.split(file_list, label_list):
        train_fold.append([pairs_list[idx] for idx in list(train_index)])
        valid_fold.append([pairs_list[idx] for idx in list(valid_index)])

    return train_fold[fold_idx], valid_fold[fold_idx]
####
def prepare_colon_manual_data(fold_idx=0):
    def load_data_info(pathname, parse_label=True, label_value=0):
        file_list = glob.glob(pathname)
        if parse_label:
            label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        else:
            label_list = [label_value for file_path in file_list]
        print(Counter(label_list))
        return list(zip(file_list, label_list))

    assert fold_idx < 5, "Currently only support 5 fold, each fold is 1 TMA"

    data_root_dir = '../../train/COLON_MANUAL_PATCHES/'
    tma_list = ['1010711', '1010712', '1010716']

    set_1010711 = load_data_info('%s/v1/1010711/*.jpg' % data_root_dir)
    set_1010712 = load_data_info('%s/v1/1010712/*.jpg' % data_root_dir)
    set_1010716 = load_data_info('%s/v1/1010716/*.jpg' % data_root_dir)
    wsi = load_data_info('%s/wsi/*.jpg' % data_root_dir, parse_label=False, label_value=0) # benign exclusively
    random.shuffle(wsi)

    train_set = set_1010711 + set_1010712 + wsi[:225]
    valid_set = set_1010716 + wsi[225:]
    return train_set, valid_set

####
def prepare_prostate_asan_data(fold_idx=0):
    assert fold_idx < 5, "Currently only support 5 fold, each fold is 1 TMA"

    tma_list = ['11S-1_1(x400)', '11S-1_2(x400)', '11S-1_3(x400)']
    class_dict = {
        'benign' : 0, '3': 1, '4' : 2, '5' : 3
    }

    file_list = []
    label_list = []
    for tma_code in tma_list:
        # tma_file_list = glob.glob('../../train/PROSTATE_ASAN/1024x1024_512x512/%s/*.jpg' % tma_code)
        tma_file_list = glob.glob('/mnt/dang/train/PROSTATE_ASAN/1024x1024_512x512/%s/*.jpg' % tma_code)
        tma_label_list = [file_path.split('_')[-1].split('.')[0] for file_path in tma_file_list]
        tma_label_list = [class_dict[label] for label in tma_label_list]
        file_list.extend(tma_file_list)
        label_list.extend(tma_label_list)
    pairs_list = list(zip(file_list, label_list))
    # [(0, 139), (1, 235), (2, 645), (3, 194)] highly imbalance

    train_fold = []
    valid_fold = []
    skf = StratifiedKFold(n_splits=5, random_state=5, shuffle=False)
    for train_index, valid_index in skf.split(file_list, label_list):
        train_fold.append([pairs_list[idx] for idx in list(train_index)])
        valid_fold.append([pairs_list[idx] for idx in list(valid_index)])

    return train_fold[fold_idx], valid_fold[fold_idx]

####
def visualize(ds, batch_size, nr_steps=100):
    data_idx = 0
    cmap = plt.get_cmap('jet')
    for i in range(0, nr_steps):
        if data_idx >= len(ds):
            data_idx = 0
        for j in range(1, batch_size+1):
            sample = ds[data_idx+j]
            if len(sample) == 2:
                img = sample[0]
            else:
                img = sample[0]
                # TODO: case with multiple channels
                aux = np.squeeze(sample[-1])
                aux = cmap(aux)[...,:3] # gray to RGB heatmap
                aux = (aux * 255).astype('uint8')
                img = np.concatenate([img, aux], axis=0)
                img = cv2.resize(img, (40, 80), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, batch_size, j)
            plt.title(str(sample[1]))
            plt.imshow(img)
        plt.show()
        data_idx += batch_size
