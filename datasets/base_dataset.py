import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from glob import glob
import numpy as np
from scipy.io import loadmat
from PIL import Image

import random
import os

from PeopleCounting.utils.data import random_crop, get_padding
from datasets.dataset_metadata import UCF_CC_50Metadata, SmartCityMetadata, ShanghaiTechAMetadata

class BaseDataset(Dataset):

    def __init__(self, dname, root, crop_size, downsample, log_para, method, split_file):
        self.dname = dname
        self.root = root
        self.crop_size = crop_size
        self.downsample = downsample
        self.log_para = log_para
        self.method = method

        if dname == 'UCF_CC_50':
            self.meta = UCF_CC_50Metadata
        elif dname == 'SmartCity':
            self.meta = SmartCityMetadata
        elif dname in ['ShainghaiTechA', 'SHHA']:
            self.meta = ShanghaiTechAMetadata
        else:
            raise NotImplementedError

        if self.meta.is_grey:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.meta.transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        img_fns = glob(os.path.join(root, self.meta.img_folder, '*.jpg'))
        if split_file is None:
            self.img_fns = img_fns
        else:
            with open(split_file, 'r') as f:
                split_fns = f.readlines()
                split_fns = [fn.strip('\n') for fn in split_fns]
            self.img_fns = [fn for fn in img_fns if fn in split_fns]
        
    def __len__(self):
        return len(self.img_fns)

    def __getitem__(self, index):
        img = self._get_image(index)
        gt = self._get_gt(index)
        
        if self.method == 'train':
            return tuple(self._train_transform(img, gt))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt))

    def _get_image(self, index):
        img_fn = self.img_fns[index]
        img = Image.open(img_fn).convert('RGB')
        return img

    def _get_gt(self, index):
        img_fn = self.img_fns[index]
        gt_fn = os.path.join(self.root, self.meta.gt_folder, 
            self.meta.gt_prefix+img_fn.split('/')[-1].split('.')[0]+self.meta.gt_suffix+'.mat')
        if self.dname in ['ShainghaiTechA', 'SHHA']:
            gt = loadmat(gt_fn)[self.meta.gt_colname][0][0][0][0][0].astype(int)
        else:
            gt = loadmat(gt_fn)[self.meta.gt_colname].astype(int)
        return gt

    def _train_transform(self, img, gt):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert len(gt) >= 0
        
        # Padding
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, ht, wd = get_padding(ht, wd, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            gt = gt + [left, top]

        # Cropping
        h, w = self.crop_size, self.crop_size

        i, j = random_crop(ht, wd, h, w)
        img = F.crop(img, i, j, h, w)

        if len(gt) > 0:
            gt = gt - [j, i]
            idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= w) * \
                       (gt[:, 1] >= 0) * (gt[:, 1] <= h)
            gt = gt[idx_mask]
        else:
            gt = np.empty([0, 2])

        # Downsampling
        gt = gt / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()

        return img, gt

    def _val_transform(self, img, gt):
        # Padding
        wd, ht = img.size
        new_wd = (wd // self.downsample + 1) * self.downsample if wd % self.downsample != 0 else wd
        new_ht = (ht // self.downsample + 1) * self.downsample if ht % self.downsample != 0 else ht

        padding, ht, wd = get_padding(ht, wd, new_ht, new_wd)
        left, top, _, _ = padding

        img = F.pad(img, padding)
        gt = gt + [left, top]

        # Downsampling
        gt = gt / self.downsample

        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()

        return img, gt