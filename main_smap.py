import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from models import Model
from losses import Loss
from datasets import Dataset
from losses.post_prob import Post_Prob
from utils.misc import AverageMeter

def train_collate(batch, dataset):
    if dataset == 'ScaleBayesian':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        targets = transposed_batch[2]
        st_sizes = torch.FloatTensor(transposed_batch[3])
        smaps = torch.stack(transposed_batch[4], 0)
        return images, points, targets, st_sizes, smaps

    elif dataset == 'Binary':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        gt_discretes = torch.stack(transposed_batch[2], 0)
        return images, points, gt_discretes

    elif dataset == 'Scale':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[2], 0)
        smaps = torch.stack(transposed_batch[3], 0)
        return images, points, dmaps, smaps
    
    else:
        raise NotImplementedError

# class JointLoss(nn.Module):
#     def __init__(self, countLoss):
#         super().__init__()
#         self.countLoss = countLoss
#         self.scaleLoss = nn.L1Loss()

#     def forward(self, countLossInput, pred_smap, gt_smap, beta):
#         count_l = self.countLoss(*countLossInput)
#         scale_l = self.scaleLoss(pred_smap, gt_smap)
#         # print('count loss: {:.4f}, scale loss: {:.4f}'.format(count_l, scale_l))
#         return count_l + beta * scale_l

class Trainer(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.model = Model(self.hparams.model_name, **self.hparams.model)
        self.countloss = Loss(self.hparams.loss_name, **self.hparams.loss)
        self.scaleloss = nn.MSELoss()
        # self.numloss = nn.L1Loss()
        self.train_dataset = Dataset(self.hparams.dataset_name, method='train', **self.hparams.dataset)
        self.val_dataset = Dataset(self.hparams.dataset_name, method='val', **self.hparams.dataset)
        self.test_dataset = Dataset(self.hparams.dataset_name, method='test', **self.hparams.dataset)

        if self.hparams.loss_name == 'Bayesian':
            self.post_prob = Post_Prob(**self.hparams.post_prob)
        elif self.hparams.loss_name == 'OT':
            self.tv_loss = nn.L1Loss(reduction='none')
            self.count_loss = nn.L1Loss()

    def train_collate_fn(self, batch):
        return train_collate(batch, self.hparams.dataset_name)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size=self.hparams.batch_size_train, 
            num_workers=self.hparams.num_workers,
            collate_fn=self.train_collate_fn,
            pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
            batch_size=self.hparams.batch_size_val, 
            num_workers=self.hparams.num_workers,
            pin_memory=True, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
            batch_size=self.hparams.batch_size_val, 
            num_workers=self.hparams.num_workers,
            pin_memory=True, shuffle=False, drop_last=False)

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        if self.hparams.dataset_name == 'ScaleBayesian':
            imgs, gts, targs, st_sizes, smaps = batch
            preds, pred_smaps = self.forward(imgs)
            prob_list = self.post_prob(gts, st_sizes)
            loss = self.loss([prob_list, targs, preds], pred_smaps, smaps)

        elif self.hparams.dataset_name == 'Binary':
            imgs, gts, bmaps = batch
            preds = self.forward(imgs)
            preds_sum = preds.view([len(imgs), -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            preds_normed = preds / (preds_sum + 1e-6)
            gd_count = torch.from_numpy(np.array([len(p) for p in gts], dtype=np.float32)).float().cuda()

            ot_loss, wd, ot_obj_value = self.loss(preds_normed, preds, gts)
            ot_loss = ot_loss * self.hparams.wot
            ot_obj_value = ot_obj_value * self.hparams.wot

            count_loss = self.count_loss(preds.sum(1).sum(1).sum(1), gd_count)

            bmaps_normed = bmaps / (gd_count.unsqueeze(1).unsqueeze(2).unsqueeze(3) + 1e-6)
            tv_loss = (self.tv_loss(preds_normed, bmaps_normed).sum(1).sum(1).sum(1) * gd_count).mean(0) * self.hparams.wtv

            loss = ot_loss + tv_loss + count_loss

        elif self.hparams.dataset_name == 'Scale':
            imgs, gts, dmaps, smaps = batch
            # e = self.current_epoch
            beta = 1.0 # np.exp2((60-e)/20)
            lamda = 1.0
            
            with torch.cuda.amp.autocast():
                pred_dmapss, pred_smaps = self.forward(imgs)
                lower_smaps = torch.clamp(torch.floor(smaps).to(torch.long), 0, 4)
                lower_scales = nn.functional.one_hot(lower_smaps, 5).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
                pred_dmaps = (pred_dmapss * lower_scales).sum(1, keepdim=True)
                countloss = self.countloss(pred_dmaps, dmaps)
                scaleloss = self.scaleloss(pred_smaps, smaps)
                # pred_nums = pred_dmaps.sum(dim=(1, 2, 3))
                # gt_nums = dmaps.sum(dim=(1, 2, 3))
                # ratio = gt_nums / (pred_nums + 1e-6)
                # numloss = self.numloss(ratio, torch.ones_like(ratio))
                # print('scale: {:.4f}, count loss: {:.4f}, scale loss: {:.4f}, beta: {:.4f}'.format(pred_smaps.sum(), countloss, scaleloss, beta))
                loss = countloss + beta * scaleloss # + lamda * numloss
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in validation'

        patch_size = self.hparams.patch_size
        if h >= patch_size or w >= patch_size:
            img_patches = []
            pred_count = 0
            h_stride = int(np.ceil(1.0 * h / patch_size))
            w_stride = int(np.ceil(1.0 * w / patch_size))
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * patch_size
                    if i != h_stride - 1:
                        h_end = (i + 1) * patch_size
                    else:
                        h_end = h
                    w_start = j * patch_size
                    if j != w_stride - 1:
                        w_end = (j + 1) * patch_size
                    else:
                        w_end = w
                    img_patches.append(img[..., h_start:h_end, w_start:w_end])
            
            for patch in img_patches:
                dmaps, smap = self.forward(patch)
                lower_smap = torch.clamp(torch.floor(smap).to(torch.long), 0, 4)
                lower_scale = nn.functional.one_hot(lower_smap, 5).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
                pred = (dmaps * lower_scale).sum(1, keepdim=True)
                    
                pred_count += torch.sum(pred).item() / self.hparams.log_para

        else:
            dmaps, smap = self.forward(img)
            lower_smap = torch.clamp(torch.floor(smap).to(torch.long), 0, 4)
            lower_scale = nn.functional.one_hot(lower_smap, 5).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
            pred = (dmaps * lower_scale).sum(1, keepdim=True)

            pred_count = torch.sum(pred).item() / self.hparams.log_para
            
        gt_count = gt.shape[1]

        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        nae = mae / gt_count

        self.log_dict({'val/MSE': mse, 'val/MAE': mae, 'val/NAE': nae})

    def test_step(self, batch, batch_idx):
        img, gt = batch
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in validation'

        patch_size = self.hparams.patch_size
        if h >= patch_size or w >= patch_size:
            img_patches = []
            pred_count = 0
            h_stride = int(np.ceil(1.0 * h / patch_size))
            w_stride = int(np.ceil(1.0 * w / patch_size))
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * patch_size
                    if i != h_stride - 1:
                        h_end = (i + 1) * patch_size
                    else:
                        h_end = h
                    w_start = j * patch_size
                    if j != w_stride - 1:
                        w_end = (j + 1) * patch_size
                    else:
                        w_end = w
                    img_patches.append(img[..., h_start:h_end, w_start:w_end])
            
            for patch in img_patches:
                dmaps, smap = self.forward(patch)
                lower_smap = torch.clamp(torch.floor(smap).to(torch.long), 0, 4)
                lower_scale = nn.functional.one_hot(lower_smap, 5).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
                pred = (dmaps * lower_scale).sum(1, keepdim=True)
                    
                pred_count += torch.sum(pred).item() / self.hparams.log_para

        else:
            dmaps, smap = self.forward(img)
            lower_smap = torch.clamp(torch.floor(smap).to(torch.long), 0, 4)
            lower_scale = nn.functional.one_hot(lower_smap, 5).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
            pred = (dmaps * lower_scale).sum(1, keepdim=True)

            pred_count = torch.sum(pred).item() / self.hparams.log_para
            
        gt_count = gt.shape[1]

        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        nae = mae / gt_count

        self.log_dict({'test/MSE': mse, 'test/MAE': mae, 'test/NAE': nae})

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-9, verbose=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        # return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val/MAE'}
        # return {'optimizer': optimizer}

if __name__ == '__main__':
    cli = LightningCLI(Trainer, save_config_overwrite=True)