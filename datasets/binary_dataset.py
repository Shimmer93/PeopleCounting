import numpy as np
import torch
import torchvision.transforms.functional as F

import random

from datasets.base_dataset import BaseDataset
from utils.image import random_crop, add_margin

class BinaryMapDataset(BaseDataset):

    def __init__(self, dname, root, crop_size, downsample, log_para, method, split_file):
        assert crop_size % downsample == 0

        super().__init__(dname, root, crop_size, downsample, log_para, method, split_file)

    def __getitem__(self, index):
        img = self._get_image(index)
        gt = self._get_gt(index)

        if self.method == 'train':
            return tuple(self._train_transform(img, gt))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt))

    def _gen_discrete_map(self, im_height, im_width, points):
        """
            func: generate the discrete map.
            points: [num_gt, 2], for each row: [width, height]
            """
        discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
        h, w = discrete_map.shape[:2]
        num_gt = points.shape[0]
        if num_gt == 0:
            return discrete_map
        
        # fast create discrete map
        points_np = np.array(points).round().astype(int)
        p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
        p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
        p_index = torch.from_numpy(p_h* im_width + p_w)
        discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).numpy()

        ''' slow method
        for p in points:
            p = np.round(p).astype(int)
            p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
            discrete_map[p[0], p[1]] += 1
        '''
        assert np.sum(discrete_map) == num_gt
        return discrete_map

    def _train_transform(self, img, gt):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.crop_size
        assert len(gt) >= 0
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

        gt_discrete = self._gen_discrete_map(h, w, gt)
        down_w = w // self.downsample
        down_h = h // self.downsample
        gt_discrete = gt_discrete.reshape([down_h, self.downsample, down_w, self.downsample]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(gt)

        if len(gt) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                gt[:, 0] = w - gt[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.transform(img), torch.from_numpy(gt.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()

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

        return self.transform(img), torch.from_numpy(gt.copy()).float()