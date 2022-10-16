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

class BayesianTemporalDataset(BaseTemporalDataset):

    def __init__(self, root, crop_size, seq_len, downsample, log_para, method, is_grey):
        assert crop_size % downsample == 0
        super().__init__(root, crop_size, seq_len, downsample, log_para, method, is_grey)

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        frame_id = int(os.path.basename(img_fn).split('_')[1].split('.')[0])
        frame_ids = [np.maximum(frame_id - i, 1) for i in range(self.seq_len)]
        img_fns = [img_fn.replace('_'+str(frame_id).zfill(3), '_'+str(id).zfill(3)) for id in frame_ids]
        imgs = [Image.open(fn).convert('RGB') for fn in img_fns]
        gt_fns = [fn.replace('jpg', 'npy') for fn in img_fns]
        gts = [np.load(fn) for fn in gt_fns]
        dists = [self._cal_dists(gt) for gt in gts]

        if self.method == 'train':
            return tuple(self._train_transform(imgs, gts, dists))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(imgs, gts))

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

    def _train_transform(self, imgs, gts, dists):
        w, h = imgs[0].size

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
            gts = [gt * factor for gt in gts]

        imgs = [F.to_tensor(img) for img in imgs]
        imgs = torch.stack(imgs, dim=0)
        
        # Padding
        st_size = 1.0 * min(w, h)
        if st_size < self.crop_size:
            st_size = self.crop_size
            padding, h, w = get_padding(h, w, self.crop_size, self.crop_size)
            left, top, _, _ = padding

            imgs = F.pad(imgs, padding)
            gts = [gt + [left, top] for gt in gts]

        # Cropping
        i, j = random_crop(h, w, self.crop_size, self.crop_size)
        h, w = self.crop_size, self.crop_size
        imgs = F.crop(imgs, i, j, h, w)
        h, w = self.crop_size, self.crop_size

        targs = []
        for i, gt in enumerate(gts):
            if len(gt) > 0:
                nearest_dis = np.clip(dists[i], 4.0, 128.0)

                points_left_up = gt - nearest_dis / 2.0
                points_right_down = gt + nearest_dis / 2.0
                bbox = np.concatenate((points_left_up, points_right_down), axis=1)
                inner_area = cal_inner_area(j, i, j + w, i + h, bbox)
                origin_area = np.squeeze(nearest_dis * nearest_dis, axis=-1)
                ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
                mask = (ratio >= 0.3)

                targ = ratio[mask]
                gts[i] = gt[mask]
                gts[i] = gts[i] - [j, i]  # change coodinate
            else:
                targ = np.array([])
            targs.append(targ)

        # Downsampling
        gts = [gt / self.downsample for gt in gts]

        # Flipping
        if random.random() > 0.5:
            imgs = F.hflip(imgs)
        for i, gt in enumerate(gts):
            if len(gt) > 0:
                gts[i][:, 0] = w - gt[:, 0]
        
        # Post-processing
        imgs = self.transform(imgs)
        if self.channel_first:
            imgs = imgs.transpose(0, 1)
        gts = [torch.from_numpy(gt.copy()).float() for gt in gts]
        targs = [torch.from_numpy(targ.copy()).float() for targ in targs]

        return imgs, gts, targs, st_size

if __name__ == '__main__':
    dataset = BayesianTemporalDataset('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/fdst', 512, 5, 1, True, 'val', False)
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