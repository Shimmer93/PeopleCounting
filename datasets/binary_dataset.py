import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import random

from datasets.base_dataset import BaseDataset
from utils.data import random_crop, get_padding

class BinaryMapDataset(BaseDataset):

    def __init__(self, root, crop_size, downsample, log_para, method, is_grey, unit_size):
        assert crop_size % downsample == 0
        super().__init__(root, crop_size, downsample, log_para, method, is_grey, unit_size)

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img = Image.open(img_fn).convert('RGB')
        gt_fn = img_fn.replace('jpg', 'npy')
        gt = np.load(gt_fn)
        
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
        w, h = img.size
        assert len(gt) >= 0

        bmap = self._gen_discrete_map(h, w, gt)
        bmap = torch.from_numpy(bmap)

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
            bmap = F.resize(bmap, (h, w))
            gt = gt * factor
        
        # Padding
        st_size = 1.0 * min(w, h)
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, h, w = get_padding(h, w, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            bmap = F.pad(bmap, padding)
            gt = gt + [left, top]

        # Cropping
        i, j = random_crop(h, w, self.crop_size, self.crop_size)
        h, w = self.crop_size, self.crop_size
        img = F.crop(img, i, j, h, w)
        h, w = self.crop_size, self.crop_size
        bmap = F.crop(bmap, i, j, h, w)
        h, w = self.crop_size, self.crop_size

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
        bmap = torch.unsqueeze(bmap, 0).float()

        return img, gt, bmap