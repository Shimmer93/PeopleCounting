import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from glob import glob
import cv2
import numpy as np
from scipy.io import loadmat
from PIL import Image

import random
import os

class BaseDataset(Dataset):

    def __init__(self, dname, root, crop_size, downsample, log_para, method, split_file):
        self.dname = dname
        self.root = root
        self.crop_size = crop_size
        self.downsample = downsample
        self.log_para = log_para
        self.method = method

        if dname == 'UCF_CC_50':
            self.img_folder = ''
            self.gt_folder = ''
            self.gt_colname = 'annPoints'
            self.gt_suffix = '_ann'
            self.is_grey = True
        elif dname == 'SmartCity':
            self.img_folder = 'images'
            self.gt_folder = 'images'
            self.gt_colname = 'loc'
            self.gt_suffix = ''
            self.is_grey = False
        else:
            raise NotImplementedError

        if self.is_grey:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        img_fns = glob(os.path.join(root, self.img_folder, '*.jpg'))
        with open(split_file, 'r') as f:
            split_fns = f.readlines()
            split_fns = [fn.strip('\n') for fn in split_fns]
        self.img_fns = [fn for fn in img_fns if fn in split_fns]
        
    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, index):
        img = self._get_image(index)
        gt = self._get_gt(index)
        return (self.trans(img), torch.from_numpy(gt.copy()).float())

    def _get_image(self, index):
        img_fn = self.img_fns[index]
        img = Image.open(img_fn).convert('RGB')
        return img

    def _get_gt(self, index):
        img_fn = self.img_fns[index]
        gt_fn = os.path.join(self.root, self.gt_folder, img_fn.split('/')[-1].split('.')[0]+self.gt_suffix+'.mat')
        gt = loadmat(gt_fn)[self.gt_colname].astype(int)
        return gt