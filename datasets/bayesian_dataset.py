import numpy as np
import torch
import torchvision.transforms.functional as F

import random

from datasets.base_dataset import BaseDataset
from utils.data import random_crop, cal_inner_area, get_padding

class BayesianDataset(BaseDataset):

    def __init__(self, dname, root, crop_size, downsample, log_para, method, split_file):
        assert crop_size % downsample == 0

        super().__init__(dname, root, crop_size, downsample, log_para, method, split_file)
    
    def __getitem__(self, index):
        img = self._get_image(index)
        gt = self._get_gt(index)
        dists = self._cal_dists(gt)

        if self.method == 'train':
            return tuple(self._train_transform(img, gt, dists))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt))

    def _cal_dists(self, pts):
        square = np.sum(pts*pts, axis=1)
        dists = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(pts, pts.T) + square[None, :], 0.0))
        if len(pts) == 1:
            return np.array([[4.0]])
        elif len(pts) < 4:
            return np.mean(dists[:,1:], axis=1, keepdims=True)
        dists = np.mean(np.partition(dists, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
        return dists

    def _train_transform(self, img, gt, dists):
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
            if len(gt) > 1:
                nearest_dis = np.clip(dists, 4.0, 128.0)
            else:
                nearest_dis = np.array([4.0])

            points_left_up = gt - nearest_dis / 2.0
            points_right_down = gt + nearest_dis / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_inner_area(j, i, j + w, i + h, bbox)
            origin_area = np.squeeze(nearest_dis * nearest_dis, axis=-1)
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            target = ratio[mask]
            gt = gt[mask]

        if len(gt) > 0:
            gt = gt - [j, i]  # change coodinate
        else:
            gt = np.empty([0, 2])

        # Downsampling
        gt = gt / self.downsample
        st_size = st_size / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
            else:
                target = np.array([])
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        target = torch.from_numpy(target.copy()).float()

        return img, gt, target, st_size