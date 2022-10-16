import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import random
import os

import sys
sys.path.append('/mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting')

from datasets.base_dataset import BaseDataset
from utils.data import random_crop, get_padding

class DensityMapDataset(BaseDataset):

    def __init__(self, root, crop_size, downsample, log_para, method, is_grey):
        assert crop_size % downsample == 0
        super().__init__(root, crop_size, downsample, log_para, method, is_grey)
    
    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img = Image.open(img_fn).convert('RGB')
        gt_fn = img_fn.replace('jpg', 'npy')
        gt = np.load(gt_fn)
        basename = os.path.basename(img_fn).replace('.jpg', '')
        dmap_fn = gt_fn.replace(basename, basename + '_dmap')
        dmap = np.load(dmap_fn)

        if self.method == 'train':
            return tuple(self._train_transform(img, gt, dmap))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt))

    def _train_transform(self, img, gt, dmap):
        w, h = img.size
        assert len(gt) >= 0

        dmap = torch.from_numpy(dmap).unsqueeze(0)

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
            dmap = F.resize(dmap, (h, w))
            if len(gt) > 0:
                gt = gt * factor
        
        # Padding
        st_size = 1.0 * min(w, h)
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, h, w = get_padding(h, w, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            dmap = F.pad(dmap, padding)
            if len(gt) > 0:
                gt = gt + [left, top]

        # Cropping
        i, j = random_crop(h, w, self.crop_size, self.crop_size)
        h, w = self.crop_size, self.crop_size
        img = F.crop(img, i, j, h, w)
        h, w = self.crop_size, self.crop_size
        dmap = F.crop(dmap, i, j, h, w)
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
        dmap = dmap.reshape([1, down_h, self.downsample, down_w, self.downsample]).sum(dim=(2, 4))

        if len(gt) > 0:
            gt = gt / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            dmap = F.hflip(dmap)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        dmap = dmap.float()

        return img, gt, dmap

if __name__ == '__main__':
    ds = DensityMapDataset('UCF_CC_50', '/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/UCF_CC_50',
            '/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/UCF_CC_50/dmaps', 328, 8, 1, 'train', 
            '/mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/data/UCF_CC_50/train.txt')

    img, gt, dmap = ds[0]

    print(img.shape, dmap.shape)
    print(torch.sum(dmap), len(gt))