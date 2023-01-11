import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import numpy as np
import matplotlib.pyplot as plt
import os

from models import Model
from losses import Loss
from datasets import Dataset
from losses.post_prob import Post_Prob
from utils.misc import AverageMeter
from utils.data import divide_img_into_patches, denormalize

def train_collate(batch, dataset):
    if dataset == 'Bayesian':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        targets = transposed_batch[2]
        st_sizes = torch.FloatTensor(transposed_batch[3])
        return images, points, targets, st_sizes

    elif dataset == 'Binary':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        gt_discretes = torch.stack(transposed_batch[2], 0)
        return images, points, gt_discretes

    elif dataset == 'Density':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[2], 0)
        return images, points, dmaps
    
    else:
        raise NotImplementedError

class Trainer(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.model = Model(self.hparams.model_name, **self.hparams.model)
        # state_dict = torch.load('/mnt/home/zpengac/USERDIR/Crowd_counting/Boosting-Crowd-Counting-via-Multifaceted-Attention/weights/jhu.pth')
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[6:]
        #     new_state_dict[name] = v
        # self.model.load_state_dict(new_state_dict)
        # self.model.load_state_dict(state_dict)
        self.loss = Loss(self.hparams.loss_name, **self.hparams.loss)
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
            pin_memory=False, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
            batch_size=self.hparams.batch_size_val, 
            num_workers=self.hparams.num_workers,
            pin_memory=False, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
            batch_size=self.hparams.batch_size_val, 
            num_workers=self.hparams.num_workers,
            pin_memory=False, shuffle=False, drop_last=False)

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        if self.hparams.dataset_name == 'Bayesian':
            imgs, gts, targs, st_sizes = batch
            preds = self.forward(imgs)
            prob_list = self.post_prob(gts, st_sizes)
            if self.hparams.model_name == 'MAN':
                outputs, features = preds
                loss = self.loss(prob_list, targs, outputs)
                loss_c = 0
                for feature in features:
                    mean_feature = torch.mean(feature, dim=0)
                    mean_sum = torch.sum(mean_feature**2)**0.5
                    cosine = 1 - torch.sum(feature*mean_feature, dim=1) / (mean_sum * torch.sum(feature**2, dim=1)**0.5 + 1e-5)
                    loss_c += torch.sum(cosine)
                loss += loss_c
            else:
                loss = self.loss(prob_list, targs, preds)

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

        elif self.hparams.dataset_name == 'Density':
            imgs, gts, dmaps = batch
            if self.hparams.model_name == 'DiffusionCounter':
                loss = self.model(imgs, dmaps)
            else:
                with torch.cuda.amp.autocast():
                    preds = self.forward(imgs)
                    loss = self.loss(preds, dmaps * self.hparams.log_para)
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt, name = batch
        b, _, h, w = img.shape

        assert b == 1, 'batch size should be 1 in validation'

        patch_size = self.hparams.patch_size
        if h >= patch_size or w >= patch_size:
            pred_count = 0
            img_patches = divide_img_into_patches(img, patch_size)
            
            for patch in img_patches:
                if self.hparams.model_name == 'DiffusionCounter':
                    pred = self.model.sample(patch, 1)
                else:
                    # print(patch.shape)
                    pred = self.forward(patch)
                    if self.hparams.model_name == 'MAN':
                        pred, _ = pred
                pred_count += torch.sum(pred).item() / self.hparams.log_para

        else:
            if self.hparams.model_name == 'DiffusionCounter':
                pred = self.model.sample(patch, 1)
            else:
                # print(img.shape)
                pred = self.forward(img)
                if self.hparams.model_name == 'MAN':
                    pred, _ = pred
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
                pred = self.forward(patch)
                if self.hparams.model_name == 'MAN':
                    pred, _ = pred
                pred_count += torch.sum(pred).item() / self.hparams.log_para
                hi = i//nw
                wi = i%nw
                _, _, patch_h, patch_w = patch.shape
                pred_dmap[..., hi*patch_h//downsample:(hi+1)*patch_h//downsample, wi*patch_w//downsample:(wi+1)*patch_w//downsample] = pred / self.hparams.log_para
                
        else:
            pred = self.forward(img)
            if self.hparams.model_name == 'MAN':
                pred, _ = pred
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