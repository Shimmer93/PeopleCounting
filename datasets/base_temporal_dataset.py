import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

import os
import random

from datasets.base_dataset import BaseDataset
from utils.data import random_crop, cal_inner_area, get_padding

class BaseTemporalDataset(BaseDataset):

    def __init__(self, root, crop_size, seq_len, downsample, log_para, method, is_grey, unit_size, channel_first=True):
        super().__init__(root, crop_size, downsample, log_para, method, is_grey, unit_size)

        self.seq_len = seq_len
        self.channel_first = channel_first

        if self.is_grey:
            self.transform = T.Compose([
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        frame_id = int(os.path.basename(img_fn).split('_')[1].split('.')[0])
        frame_ids = [np.maximum(frame_id - i, 1) for i in range(self.seq_len)]
        img_fns = [img_fn.replace('_'+str(frame_id).zfill(3), '_'+str(id).zfill(3)) for id in frame_ids]
        imgs = [Image.open(fn).convert('RGB') for fn in img_fns]
        gt_fn = img_fn.replace('jpg', 'npy')
        gt = np.load(gt_fn)

        if self.method == 'train':
            return tuple(self._train_transform(imgs, gt))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(imgs, gt))

    def _train_transform(self, imgs, gt):
        w, h = imgs[0].size
        assert len(gt) >= 0

        # Grey Scale
        if random.random() > 0.88:
            imgs = [img.convert('L').convert('RGB') for img in imgs]

        # Resizing
        factor = random.random() * 0.5 + 0.75
        new_w = (int)(w * factor)
        new_h = (int)(h * factor)
        if min(new_w, new_h) >= self.crop_size:
            w = new_w
            h = new_h
            imgs = [img.resize((w, h)) for img in imgs]
            gt = gt * factor

        imgs = [F.to_tensor(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        
        # Padding
        st_size = 1.0 * min(w, h)
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, h, w = get_padding(h, w, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            imgs = F.pad(imgs, padding)
            gt = gt + [left, top]

        # Cropping
        i, j = random_crop(h, w, self.crop_size, self.crop_size)
        h, w = self.crop_size, self.crop_size
        imgs = F.crop(imgs, i, j, h, w)
        h, w = self.crop_size, self.crop_size

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
            imgs = F.hflip(imgs)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        imgs = self.transform(imgs)
        if self.channel_first:
            imgs = imgs.transpose(0, 1)
        gt = torch.from_numpy(gt.copy()).float()

        return imgs, gt

    def _val_transform(self, imgs, gts):
        if self.unit_size > 0:
            # Padding
            w, h = img.size
            new_w = (w // self.unit_size + 1) * self.unit_size if w % self.unit_size != 0 else w
            new_h = (h // self.unit_size + 1) * self.unit_size if h % self.unit_size != 0 else h

            padding, h, w = get_padding(h, w, new_h, new_w)
            left, top, _, _ = padding

            imgs = [F.to_tensor(img) for img in imgs]
            imgs = torch.stack(imgs, dim=0)

            imgs = F.pad(imgs, padding)
            gts = [(gt + [left, top] if len(gt)>0 else gt) for gt in gts]

        # Downsampling
        gts = [(gt / self.downsample if len(gt)>0 else gt) for gt in gts]

        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()

        # Post-processing
        imgs = self.transform(imgs)
        if self.channel_first:
            imgs = imgs.transpose(0, 1)
        gts = [torch.from_numpy(gt.copy()).float() for gt in gts]

        return imgs, gts