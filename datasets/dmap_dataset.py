import numpy as np
import torch
import torchvision.transforms.functional as F

import random
import os

import sys
sys.path.append('/mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting')

from datasets.base_dataset import BaseDataset
from utils.data import random_crop, get_padding

class DensityMapDataset(BaseDataset):

    def __init__(self, dname, root, dmap_path, crop_size, downsample, log_para, method, split_file):
        assert crop_size % downsample == 0

        super().__init__(dname, root, crop_size, downsample, log_para, method, split_file)
        self.dmap_path = dmap_path
    
    def __getitem__(self, index):
        img = self._get_image(index)
        gt = self._get_gt(index)
        dmap = self._get_dmap(index)

        if self.method == 'train':
            return tuple(self._train_transform(img, gt, dmap))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt))

    def _get_dmap(self, index):
        img_fn = self.img_fns[index]
        dmap_fn = os.path.join(self.dmap_path, img_fn.split('/')[-1].split('.')[0]+'.npy')
        dmap = np.load(dmap_fn)
        return dmap

    def _cal_dists(self, pts):
        square = np.sum(pts*pts, axis=1)
        dists = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(pts, pts.T) + square[None, :], 0.0))
        if len(pts) == 1:
            return np.array([[4.0]])
        elif len(pts) < 4:
            return np.mean(dists[:,1:], axis=1, keepdims=True)
        dists = np.mean(np.partition(dists, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
        return dists

    def _train_transform(self, img, gt, dmap):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert len(gt) >= 0
        dmap = torch.from_numpy(dmap)

        # Padding
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, ht, wd = get_padding(ht, wd, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            dmap = F.pad(dmap, padding)
            gt = gt + [left, top]

        # Cropping
        h, w = self.crop_size, self.crop_size
        h2, w2 = self.crop_size, self.crop_size

        i, j = random_crop(ht, wd, h, w)
        img = F.crop(img, i, j, h, w)
        dmap = F.crop(dmap, i, j, h2, w2)

        if len(gt) > 0:
            gt = gt - [j, i]
            idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= w) * \
                       (gt[:, 1] >= 0) * (gt[:, 1] <= h)
            gt = gt[idx_mask]
        else:
            gt = np.empty([0, 2])

        # Downsampling
        down_w = w // self.downsample
        down_h = h // self.downsample
        dmap = dmap.reshape([down_h, self.downsample, down_w, self.downsample]).sum(dim=(1, 3))

        gt = gt / self.downsample

        # Flipping
        # if random.random() > 0.5:
        #     img = F.hflip(img)
        #     dmap = F.hflip(dmap)
        #     if len(gt) > 0:
        #         gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        dmap = torch.unsqueeze(dmap, 0)

        return img, gt, dmap

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

if __name__ == '__main__':
    ds = DensityMapDataset('UCF_CC_50', '/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/UCF_CC_50',
            '/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/UCF_CC_50/dmaps', 328, 8, 1, 'train', 
            '/mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/data/UCF_CC_50/train.txt')

    img, gt, dmap = ds[0]

    print(img.shape, dmap.shape)
    print(torch.sum(dmap), len(gt))