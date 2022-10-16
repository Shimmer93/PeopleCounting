import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import random

import sys
sys.path.append('/mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting')
from datasets.base_dataset import BaseDataset
from utils.data import random_crop, cal_inner_area, get_padding

class BayesianDataset(BaseDataset):
    def __init__(self, root, crop_size, downsample, log_para, method, is_grey):
        assert crop_size % downsample == 0
        super().__init__(root, crop_size, downsample, log_para, method, is_grey)

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img = Image.open(img_fn).convert('RGB')
        gt_fn = img_fn.replace('jpg', 'npy')
        gt = np.load(gt_fn)
        dists = self._cal_dists(gt)
        
        if self.method == 'train':
            return tuple(self._train_transform(img, gt, dists))
        elif self.method in ['val', 'test']:
            name = img_fn.split('/')[-1].split('.')[0]
            return self._val_transform(img, gt)

    def _cal_dists(self, pts):
        if len(pts) == 0:
            return np.array([[]])
        elif len(pts) == 1:
            return np.array([[4.0]])
        square = np.sum(pts*pts, axis=1)
        dists = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(pts, pts.T) + square[None, :], 0.0))
        if len(pts) < 4:
            return np.mean(dists[:,1:], axis=1, keepdims=True)
        dists = np.mean(np.partition(dists, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
        return dists

    def _train_transform(self, img, gt, dists):
        w, h = img.size
        assert len(gt) >= 0

        # Grey Scale
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')

        # Resizing
        factor = random.random() * 0.5 + 0.75
        new_w = (int)(w * factor)
        new_h = (int)(h * factor)
        if min(new_w, new_h) >= self.crop_size:
            w = new_w
            h = new_h
            img = img.resize((w, h))
            gt = gt * factor
        
        # Padding
        st_size = min(w, h)
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, h, w = get_padding(h, w, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            gt = gt + [left, top]

        # Cropping
        i, j = random_crop(h, w, self.crop_size, self.crop_size)
        h, w = self.crop_size, self.crop_size
        img = F.crop(img, i, j, h, w)
        h, w = self.crop_size, self.crop_size

        if len(gt) > 0:
            nearest_dis = np.clip(dists, 4.0, 128.0)

            points_left_up = gt - nearest_dis / 2.0
            points_right_down = gt + nearest_dis / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_inner_area(j, i, j + w, i + h, bbox)
            origin_area = np.squeeze(nearest_dis * nearest_dis, axis=-1)
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            targ = ratio[mask]
            gt = gt[mask]
            gt = gt - [j, i]  # change coodinate

        # Downsampling
        #gt = gt / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
        if len(gt) > 0:
            gt[:, 0] = w - gt[:, 0]
        else:
            targ = np.array([])
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        targ = torch.from_numpy(targ.copy()).float()

        return img, gt, targ, st_size

if __name__ == '__main__':
    dataset = BayesianDataset('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/jhu', 512, 512, 1, 'val', False)
    for img, gt in dataset:
        print(img.shape)