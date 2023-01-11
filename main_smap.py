import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import numpy as np
import matplotlib.pyplot as plt
import os

import warnings
warnings.filterwarnings('ignore')

from models import Model
from losses import Loss
from datasets import Dataset
from losses.post_prob import Post_Prob_with_Scale
from utils.data import divide_img_into_patches, denormalize
from losses.prl import PRL

def train_collate(batch, dataset):
    if dataset == 'ScaleBayesian':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        targets = transposed_batch[2]
        st_sizes = torch.FloatTensor(transposed_batch[3])
        smaps = torch.stack(transposed_batch[4], 0)
        scales =  transposed_batch[5]
        return images, points, targets, st_sizes, smaps, scales

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

    elif dataset == 'Scale2':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        dmapss = [[],[],[]]
        for j, dmaps in enumerate(transposed_batch[2]):
            for i in range(3):
                dmapss[i].append(dmaps[i])
        dmapss = [torch.stack(dmapss[i], 0) for i in range(3)]
        smaps = torch.stack(transposed_batch[3], 0)
        return images, points, dmapss, smaps

    elif dataset == 'ScaleSelect':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[2], 0)
        smaps = torch.stack(transposed_batch[3], 0)
        return images, points, dmaps, smaps
    
    else:
        raise NotImplementedError

class Trainer(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.model = Model(self.hparams.model_name, **self.hparams.model)
        self.loss = Loss(self.hparams.loss_name, **self.hparams.loss)
        self.densityLoss = Loss(self.hparams.loss_name, **self.hparams.loss)
        self.scaleLoss = nn.MSELoss() #PRL(method='MSE') # nn.MSELoss()
        self.countLoss = nn.L1Loss()
        self.train_dataset = Dataset(self.hparams.dataset_name, method='train', **self.hparams.dataset)
        self.val_dataset = Dataset(self.hparams.dataset_name, method='val', **self.hparams.dataset)
        self.test_dataset = Dataset(self.hparams.dataset_name, method='test', **self.hparams.dataset)

        if self.hparams.loss_name == 'Bayesian':
            self.post_prob = Post_Prob_with_Scale(**self.hparams.post_prob)
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
            if self.current_epoch == 60:
                self.model.teacherforcing = False

            imgs, gts, targs, st_sizes, smaps, scales = batch
            pred_dmaps, pred_smaps = self.forward((imgs, smaps))
            prob_list = self.post_prob(gts, scales, st_sizes)
            densityloss = self.densityLoss(prob_list, targs, pred_dmaps)
            pred_nums = pred_dmaps.sum(dim=(1, 2, 3))
            gt_nums = torch.tensor([len(gt) for gt in gts]).float().to(pred_nums.device)
            scaleloss = self.scaleLoss(pred_smaps, smaps * 10)

            ratio = gt_nums / (pred_nums + 1e-3)
            countloss = self.countLoss(ratio, torch.ones_like(ratio))

            beta = 1.0
            lamda = 1.0
            # print('densityloss: {:.4f}, scaleloss: {:.4f}, countloss: {:.4f}'.format(densityloss, scaleloss, countloss))
            loss = densityloss + beta * scaleloss + lamda * countloss

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
            # if self.current_epoch == 40:
            #     self.model.teacherforcing = False

            imgs, gts, dmaps, smaps = batch

            alpha = 1.0
            beta = 1.0
            lamda = 0.0001
            
            pred_dmaps, pred_smaps, _ = self.forward((imgs, smaps))

            lower_smaps = torch.clamp(torch.floor(smaps).to(torch.long), 4, 6) - 4
            aug_maps = torch.nn.functional.one_hot(lower_smaps, 3).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
            aug_maps[:, 0, ...] *= 1
            aug_maps[:, 1, ...] *= 4
            aug_maps[:, 2, ...] *= 16
            aug_maps = aug_maps.sum(dim=1, keepdim=True)
            red_maps = torch.nn.functional.one_hot(lower_smaps, 3).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)
            red_maps[:, 0, ...] *= 1
            red_maps[:, 1, ...] *= 1/4
            red_maps[:, 2, ...] *= 1/16
            red_maps = red_maps.sum(dim=1, keepdim=True)

            densityloss = self.densityLoss(pred_dmaps, dmaps * aug_maps * self.hparams.log_para)
            scaleloss = self.scaleLoss(pred_smaps, smaps * 10.0)

            pred_nums = (pred_dmaps * red_maps).sum(dim=(1, 2, 3))
            # gt_nums = dmaps.sum(dim=(1, 2, 3)) * self.hparams.log_para
            gt_nums = torch.tensor([len(gt) for gt in gts]).float().to(pred_nums.device) * self.hparams.log_para
            gt_nums[gt_nums == 0] = 1
            # ratio = pred_nums / gt_nums
            # countloss = self.countLoss(ratio, torch.ones_like(ratio))
            countloss = self.countLoss(pred_nums, gt_nums)

            loss = alpha * densityloss + beta * scaleloss # + lamda * countloss

        elif self.hparams.dataset_name == 'Scale2':
            # if self.current_epoch == 60:
            #     self.model.teacherforcing = False

            imgs, gts, dmapss, smaps = batch

            # with torch.cuda.amp.autocast():
            pred_dmapss, pred_smaps = self.forward((imgs, smaps))
            dmapss = [dmaps * self.hparams.log_para for dmaps in dmapss]
            loss = self.loss(pred_dmapss, pred_smaps, dmapss, smaps * 10.0)

        elif self.hparams.dataset_name == 'ScaleSelect':
            imgs, gts, dmaps, smaps = batch

            pred_dmaps, pred_smaps = self.forward(imgs)
            loss = self.loss(pred_dmaps, pred_smaps, dmaps, smaps)

        return loss

    def validation_step(self, batch, batch_idx):
        img, gt, _ = batch
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in validation'

        patch_size = self.hparams.patch_size
        if h >= patch_size or w >= patch_size:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, patch_size)
            
            for patch in img_patches:
                pred, _, _ = self.forward(patch)
                pred_count += torch.sum(pred).item() / self.hparams.log_para

        else:
            pred, _, _ = self.forward(img)
            pred_count = torch.sum(pred).item() / self.hparams.log_para
            
        gt_count = gt.shape[1]

        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        nae = mae / gt_count if gt_count > 0 else mae

        self.log_dict({'val/MSE': mse, 'val/MAE': mae, 'val/NAE': nae})

    def test_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.test_pred_path = os.path.join(self.logger.log_dir, 'preds')
            os.makedirs(self.test_pred_path, exist_ok=True)
            print(self.test_pred_path)

        img, gt, name = batch
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in testing'

        patch_size = self.hparams.patch_size
        downsample = self.hparams.dataset['downsample']
        
        if h >= patch_size or w >= patch_size:
            pred_count = 0
            img_patches, nh, nw = divide_img_into_patches(img, patch_size)
            
            pred_dmap = torch.zeros(1,1,h//downsample, w//downsample)
            for i, patch in enumerate(img_patches):
                pred, _, _ = self.forward(patch)
                pred_count += torch.sum(pred).item() / self.hparams.log_para
                hi = i//nw
                wi = i%nw
                _, _, patch_h, patch_w = patch.shape
                pred_dmap[..., hi*patch_h//downsample:(hi+1)*patch_h//downsample, wi*patch_w//downsample:(wi+1)*patch_w//downsample] = pred / self.hparams.log_para
                
        else:
            pred, _, _ = self.forward(img)
            pred_count = torch.sum(pred).item() / self.hparams.log_para
            pred_dmap = pred / self.hparams.log_para
            
        gt_count = gt.shape[1]
        
        denormalized_img = denormalize(img)
        
        fig = plt.figure(figsize=(20, 10))
        ax_img = fig.add_subplot(1, 2, 1)
        ax_img.imshow(denormalized_img.squeeze().permute(1, 2, 0).cpu().numpy())
        ax_img.set_title(f'Name: {name[0]}')
        ax_dmap = fig.add_subplot(1, 2, 2)
        ax_dmap.imshow(pred_dmap.squeeze().detach().cpu().numpy())
        ax_dmap.set_title(f'GT: {gt_count}, Pred: {pred_count:.2f}')
        
        plt.savefig(os.path.join(self.test_pred_path, f'{name[0]}.png'))
        plt.clf()
        plt.close()

        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        nae = mae / gt_count if gt_count > 0 else mae

        if gt_count <= 10:
            add_mse = ('test/MSE_0_10', mse)
            add_mae = ('test/MAE_0_10', mae)
            add_nae = ('test/NAE_0_10', nae)
        elif gt_count <= 100:
            add_mse = ('test/MSE_11_100', mse)
            add_mae = ('test/MAE_11_100', mae)
            add_nae = ('test/NAE_11_100', nae)
        elif gt_count <= 1000:
            add_mse = ('test/MSE_101_1000', mse)
            add_mae = ('test/MAE_101_1000', mae)
            add_nae = ('test/NAE_101_1000', nae)
        else:
            add_mse = ('test/MSE_1001_inf', mse)
            add_mae = ('test/MAE_1001_inf', mae)
            add_nae = ('test/NAE_1001_inf', nae)

        self.log_dict({'test/MSE': mse, 'test/MAE': mae, 'test/NAE': nae, add_nae[0]: add_nae[1], add_mae[0]: add_mae[1], add_mse[0]: add_mse[1]})

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