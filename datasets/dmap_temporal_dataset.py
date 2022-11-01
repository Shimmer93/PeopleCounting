import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import os
import random

import sys
sys.path.append('/mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting')
from datasets.base_temporal_dataset import BaseTemporalDataset
from utils.data import random_crop, cal_inner_area, get_padding

class DensityTemporalDataset(BaseTemporalDataset):

    def __init__(self, root, crop_size, seq_len, downsample, log_para, method, is_grey, unit_size, channel_first, empty_ratio=0.0):
        assert crop_size % downsample == 0
        super().__init__(root, crop_size, seq_len, downsample, log_para, method, is_grey, unit_size, channel_first)
        self.empty_ratio = empty_ratio

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        frame_id = int(os.path.basename(img_fn).split('_')[1].split('.')[0])
        frame_ids = [np.maximum(frame_id - i, 1) for i in range(self.seq_len)]
        img_fns = [img_fn.replace('_'+str(frame_id).zfill(3), '_'+str(id).zfill(3)) for id in frame_ids]
        imgs = [Image.open(fn).convert('RGB') for fn in img_fns]
        gt_fns = [fn.replace('jpg', 'npy') for fn in img_fns]
        gts = [np.load(fn) for fn in gt_fns]
        basenames = [os.path.basename(img_fn).replace('.jpg', '') for img_fn in img_fns]
        dmap_fns = [gt_fn.replace(basename, basename + '_dmap') for gt_fn, basename in zip(gt_fns, basenames)]
        dmaps = [np.load(fn) for fn in dmap_fns]
        dmaps = np.stack(dmaps, axis=0)

        if self.method == 'train':
            return tuple(self._train_transform(imgs, gts, dmaps))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(imgs, gts))


    def _train_transform(self, imgs, gts, dmaps):
        w, h = imgs[0].size

        dmaps = torch.from_numpy(dmaps).unsqueeze(1)

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
            dmaps = F.resize(dmaps, (h, w))
            gts = [gt * factor for gt in gts]

        imgs = [F.to_tensor(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)

        # Making Empty
        if random.random() < self.empty_ratio:
            imgs[0] = torch.zeros_like(imgs[0])
            gts[0] = np.empty([0, 2])
            dmaps[0] = torch.zeros_like(dmaps[0])

        # Padding
        # st_size = 1.0 * min(w, h)
        # if st_size < self.crop_size:
        #     st_size = self.crop_size
        #     padding, h, w = get_padding(h, w, self.crop_size, self.crop_size)
        #     left, top, _, _ = padding

        #     imgs = F.pad(imgs, padding)
        #     dmaps = F.pad(dmaps, padding)
        #     gts = [(gt + [left, top] if len(gt)>0 else gt) for gt in gts]

        # Cropping
        i, j = random_crop(h, w, self.crop_size, self.crop_size)
        h, w = self.crop_size, self.crop_size
        imgs_dmaps = torch.cat([imgs, dmaps], dim=1)
        imgs_dmaps = F.crop(imgs_dmaps, i, j, h, w)
        imgs = imgs_dmaps[:, :3, :, :]
        dmaps = imgs_dmaps[:, 3:, :, :]
        assert dmaps.shape[-1] == dmaps.shape[-2], f'dmaps.shape: {dmaps.shape}, imgs.shape: {imgs.shape}'
        h, w = self.crop_size, self.crop_size

        for i, gt in enumerate(gts):
            if len(gt) > 0:
                gt = gt - [j, i]
                idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= w) * \
                        (gt[:, 1] >= 0) * (gt[:, 1] <= h)
                gts[i] = gt[idx_mask]
            else:
                gts[i] = np.empty([0, 2])

        # Downsampling
        down_w = w // self.downsample
        down_h = h // self.downsample
        dmaps = dmaps.reshape([dmaps.shape[0], 1, down_h, self.downsample, down_w, self.downsample]).sum(dim=(3, 5))
        gts = [(gt / self.downsample if len(gt)>0 else gt) for gt in gts]

        # Flipping
        if random.random() > 0.5:
            imgs = F.hflip(imgs)
            dmaps = F.hflip(dmaps)
        for i, gt in enumerate(gts):
            if len(gt) > 0:
                gts[i][:, 0] = w - gt[:, 0]
        
        # Post-processing
        imgs = self.transform(imgs)
        if self.channel_first:
            imgs = imgs.transpose(0, 1)
            dmaps = dmaps.transpose(0, 1)
        gts = [torch.from_numpy(gt.copy()).float() for gt in gts]

        return imgs, gts, dmaps

if __name__ == '__main__':
    dataset = DensityTemporalDataset('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/fdst', 512, 5, 1, True, 'val', False)
    # print(len(dataset))
    # imgs, gts, targs, st_size = dataset[0]
    # print(imgs.shape, len(gts), len(targs), st_size)
    # for gt in gts:
    #     print(gt.shape)
    # for targ in targs:
    #     print(targ.shape)

    from torch.utils.data import DataLoader

    def val_collate(batch):
        imgs, gts = zip(*batch)
        imgs = torch.stack(imgs, dim=0)
        return imgs, gts

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=val_collate)
    for imgs, gts in dataloader:
        print(imgs.shape, len(gts))
        for gt in gts:
            print(gt.shape)
        break