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

class ScaleMapDataset(BaseDataset):

    def __init__(self, root, crop_size, downsample, log_para, method, is_grey, unit_size, type=1, scale_level=8, model_level=5):
        assert crop_size % downsample == 0
        super().__init__(root, crop_size, downsample, log_para, method, is_grey, unit_size)
        self.type = type
        self.scale_level = scale_level
        self.model_level = model_level
    
    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img = Image.open(img_fn).convert('RGB')
        gt_fn = img_fn.replace('jpg', 'npy')
        gt = np.load(gt_fn)
        basename = os.path.basename(img_fn).replace('.jpg', '')
        if self.method == 'train':
            if self.type == 1:
                dmap_fn = gt_fn.replace(basename, basename + '_dmap')
            else:
                dmap_fn = gt_fn.replace(basename, basename + '_dmap2')
            dmap = np.load(dmap_fn)
            # smap_fn = gt_fn.replace(basename, basename + '_smap')
            # smap = np.load(smap_fn)
            smap_fn = gt_fn.replace(basename, basename + '_smap').replace('.npy', '.png')
            smap = np.asarray(Image.open(smap_fn)).astype(np.float32)

            return tuple(self._train_transform(img, gt, dmap, smap))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt, basename))

    def _train_transform(self, img, gt, dmap, smap):
        w, h = img.size
        assert len(gt) >= 0

        dmap = torch.from_numpy(dmap).unsqueeze(0)
        smap = torch.from_numpy(smap).unsqueeze(0)

        # Grey Scale
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')

        # # Resizing
        # factor = random.random() * 0.5 + 0.75
        # new_w = (int)(w * factor)
        # new_h = (int)(h * factor)
        # if min(new_w, new_h) >= self.crop_size:
        #     w = new_w
        #     h = new_h
        #     img = img.resize((w, h))
        #     dmap = F.resize(dmap, (h, w))
        #     if len(gt) > 0:
        #         gt = gt * factor
        
        # Padding
        st_size = 1.0 * min(w, h)
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, h, w = get_padding(h, w, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            dmap = F.pad(dmap, padding)
            smap = F.pad(smap, padding)
            if len(gt) > 0:
                gt = gt + [left, top]

        # Cropping
        i, j = random_crop(h, w, self.crop_size, self.crop_size)
        h, w = self.crop_size, self.crop_size
        img = F.crop(img, i, j, h, w)
        h, w = self.crop_size, self.crop_size
        dmap = F.crop(dmap, i, j, h, w)
        h, w = self.crop_size, self.crop_size
        smap = F.crop(smap, i, j, h, w)
        h, w = self.crop_size, self.crop_size

        if len(gt) > 0:
            gt = gt - [j, i]
            idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= w) * \
                       (gt[:, 1] >= 0) * (gt[:, 1] <= h)
            gt = gt[idx_mask]
        else:
            gt = np.empty([0, 2])

        # Add black border
        # if random.random() > 0.75:
        #     border_sizes = np.random.random_integers(0, self.downsample-1, 4).tolist()
        #     # print(border_sizes)
        #     pad_h = border_sizes[1] + border_sizes[3]
        #     pad_w = border_sizes[0] + border_sizes[2]
        #     new_h = h - pad_h
        #     new_w = w - pad_w
        #     img = F.crop(img, 0, 0, new_h, new_w)
        #     dmap = F.crop(dmap, 0, 0, new_h, new_w)
        #     smap = F.crop(smap, 0, 0, new_h, new_w)
        #     img = F.pad(img, border_sizes)
        #     dmap = F.pad(dmap, border_sizes)
        #     smap = F.pad(smap, border_sizes)

        #     if len(gt) > 0:
        #         idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= new_w) * \
        #                    (gt[:, 1] >= 0) * (gt[:, 1] <= new_h)
        #         gt = gt[idx_mask]
        #         gt = gt + [border_sizes[1], border_sizes[0]]

        # Downsampling
        down_w = w // self.downsample
        down_h = h // self.downsample
        dmap = dmap.reshape([1, down_h, self.downsample, down_w, self.downsample]).sum(dim=(2, 4))
        smap = F.resize(smap, (down_h, down_w), interpolation=Image.NEAREST)
        # smap = torch.nn.functional.max_pool2d(smap, self.downsample)

        if len(gt) > 0:
            gt = gt / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            dmap = F.hflip(dmap)
            smap = F.hflip(smap)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        dmap = dmap.float()
        smap = smap.float()

        mask = smap<=0
        smap[mask] = 1.
        smap = self.scale_level + torch.log2(smap/self.crop_size)
        # smap[mask] = 0
        # smap = torch.floor(smap).float()

        return img, gt, dmap, smap

if __name__ == '__main__':
    ds = ScaleMapDataset('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/jhu', 512, 16, 1, 'train', False, 16, 1, 8)
    for i in range(100):
        img, gt, dmap, smap = ds[i]
        smap = torch.floor(smap)
        smap = torch.flatten(smap)
        smap = smap.numpy()
        print(np.unique(smap))