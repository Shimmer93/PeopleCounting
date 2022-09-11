import numpy as np
import torch
import torchvision.transforms.functional as F

import random

from datasets.base_dataset import BaseDataset
from utils.image import random_crop, cal_inner_area, add_margin

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
        dists = np.mean(np.partition(dists, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
        return dists

    def _train_transform(self, img, gt, dists):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        # assert len(keypoints) > 0
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd*re_size)
        htt = (int)(ht*re_size)
        if min(wdd, htt) >= self.crop_size:
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            gt = gt*re_size
        st_size = min(wd, ht)
        assert st_size >= self.crop_size
        i, j = random_crop(ht, wd, self.crop_size, self.crop_size)
        h, w = self.crop_size, self.crop_size
        img = F.crop(img, i, j, h, w)
        if len(gt) > 0:
            nearest_dis = np.clip(dists, 4.0, 128.0)

            points_left_up = gt - nearest_dis[:, None] / 2.0
            points_right_down = gt + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_inner_area(j, i, j + w, i + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            target = ratio[mask]
            gt = gt[mask]
            gt = gt - [j, i]  # change coodinate
        if len(gt) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt[:, 0] = w - gt[:, 0]
        else:
            target = np.array([])
            if random.random() > 0.5:
                img = F.hflip(img)

        #gt /= self.downsample

        return self.trans(img), torch.from_numpy(gt.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size

    def _val_transform(self, img, gt):
        wd, ht = img.size
        new_wd = (wd // self.downsample + 1) * self.downsample if wd % self.downsample != 0 else wd
        new_ht = (ht // self.downsample + 1) * self.downsample if ht % self.downsample != 0 else ht

        if not (new_wd == wd and new_ht == ht):
            dw = new_wd - wd
            dh = new_ht - ht
            left = dw // 2
            right = dw // 2 + dw % 2
            top = dh // 2
            bottom = dh // 2 + dh % 2

            img = add_margin(img, top, right, bottom, left, (0, 0, 0))
            gt[:, 0] += left
            gt[:, 1] += top

        #gt /= self.downsample

        return self.trans(img), torch.from_numpy(gt.copy()).float()
                