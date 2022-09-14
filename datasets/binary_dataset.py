import numpy as np
import torch
import torchvision.transforms.functional as F

import random

from datasets.base_dataset import BaseDataset
from utils.data import random_crop, get_padding

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
        assert len(gt) >= 0
        bmap = self._gen_discrete_map(ht, wd, gt)
        bmap = torch.from_numpy(bmap)

        # Padding
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, ht, wd = get_padding(ht, wd, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            bmap = F.pad(bmap, padding)
            gt = gt + [left, top]

        # Cropping
        h, w = self.crop_size, self.crop_size
        h2, w2 = self.crop_size, self.crop_size

        i, j = random_crop(ht, wd, h, w)
        img = F.crop(img, i, j, h, w)
        bmap = F.crop(bmap, i, j, h2, w2)

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
        bmap = bmap.reshape([down_h, self.downsample, down_w, self.downsample]).sum(dim=(1, 3))

        gt = gt / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            bmap = F.hflip(bmap)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        bmap = torch.unsqueeze(bmap, 0)

        return img, gt, bmap