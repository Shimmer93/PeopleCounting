import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

from models import Model
from losses import Loss
from datasets import Dataset
from losses.post_prob import Post_Prob
from utils.misc import AverageMeter

def train_collate(batch, dataset):
    if dataset in ['Bayesian', 'BayesianTemporal']:
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

def val_collate(batch, dataset):
    if dataset == 'BayesianTemporal':
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        return images, points
    else:
        return default_collate(batch)

class Trainer(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.model = Model(self.hparams.model_name, **self.hparams.model)
        self.loss = Loss(self.hparams.loss_name, **self.hparams.loss)
        self.train_dataset = Dataset(self.hparams.dataset_name, method='train', **self.hparams.dataset)
        self.val_dataset = Dataset(self.hparams.dataset_name, method='val', **self.hparams.dataset)
        self.test_dataset = Dataset(self.hparams.dataset_name, method='test', **self.hparams.dataset)

        if self.hparams.loss_name == 'Bayesian':
            self.post_prob = Post_Prob(**self.hparams.post_prob)
        elif self.hparams.loss_name == 'OT':
            self.tv_loss = nn.L1Loss(reduction='none')
            self.count_loss = nn.L1Loss()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size=self.hparams.batch_size_train, 
            num_workers=self.hparams.num_workers,
            collate_fn=(lambda batch: train_collate(batch, self.hparams.dataset_name)),
            pin_memory=True, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
            batch_size=self.hparams.batch_size_val, 
            num_workers=self.hparams.num_workers,
            collate_fn=(lambda batch: val_collate(batch, self.hparams.dataset_name)),
            pin_memory=True, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
            batch_size=self.hparams.batch_size_val, 
            num_workers=self.hparams.num_workers,
            pin_memory=True, shuffle=False, drop_last=False)

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, batch, batch_idx):
        if self.hparams.dataset_name == 'Bayesian':
            imgs, gts, targs, st_sizes = batch
            preds = self.forward(imgs)
            prob_list = self.post_prob(gts, st_sizes)
            loss = self.loss(prob_list, targs, preds)

        elif self.hparams.dataset_name == 'BayesianTemporal':
            imgs, gts, targs, st_sizes = batch
            preds = self.forward(imgs)
            loss = 0
            for i in range(preds.shape[2]):
                chosen_gts = [gt[4*i+3] for gt in gts]
                chosen_targs = [targ[4*i+3] for targ in targs]
                prob_list = self.post_prob(chosen_gts, st_sizes)
                loss += self.loss(prob_list, chosen_targs, preds[:,:,i,...])

        elif self.hparams.dataset_name == 'Binary':
            imgs, gts, bmaps = batch
            preds = self.forward(imgs)
            preds_sum = preds.view([len(imgs), -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            preds_normed = preds / (preds_sum + 1e-6)
            gd_count = torch.from_numpy(np.array([len(p) for p in gts], dtype=np.float32)).float().to('cuda')

            ot_loss, wd, ot_obj_value = self.loss(preds_normed, preds, gts)
            ot_loss = ot_loss * self.hparams.wot
            ot_obj_value = ot_obj_value * self.hparams.wot

            count_loss = self.count_loss(preds.sum(1).sum(1).sum(1), gd_count)

            bmaps_normed = bmaps / (gd_count.unsqueeze(1).unsqueeze(2).unsqueeze(3) + 1e-6)
            tv_loss = (self.tv_loss(preds_normed, bmaps_normed).sum(1).sum(1).sum(1) * gd_count).mean(0) * self.hparams.wtv

            loss = ot_loss + tv_loss + count_loss

        elif self.hparams.dataset_name == 'Density':
            imgs, gts, dmaps = batch
            preds = self.forward(imgs)
            loss = self.loss(preds * self.hparams.log_para, dmaps * self.hparams.log_para)
        
        return loss

    def validation_step(self, batch, batch_idx):
        img, gt = batch
        if len(img.shape) == 4:
            b, _, h, w = img.shape
        else:
            b, _, _, h, w = img.shape
        assert b == 1, 'batch size should be 1 in validation'

        if h >= 512 or w >= 512:
            img_patches = []
            pred_count = 0
            h_stride = int(np.ceil(1.0 * h / 512))
            w_stride = int(np.ceil(1.0 * w / 512))
            h_step = h // h_stride
            w_step = w // w_stride
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * h_step
                    if i != h_stride - 1:
                        h_end = (i + 1) * h_step
                    else:
                        h_end = h
                    w_start = j * w_step
                    if j != w_stride - 1:
                        w_end = (j + 1) * w_step
                    else:
                        w_end = w
                    img_patches.append(img[..., h_start:h_end, w_start:w_end])
            
            for patch in img_patches:
                pred = self.forward(patch)
                if len(img.shape) == 4:
                    pred_count += torch.sum(pred).item()
                else:
                    pred_count += torch.sum(pred, dim=(0,1,3,4)).cpu().numpy()

        else:
            pred = self.forward(img)
            if len(img.shape) == 4:
                pred_count = torch.sum(pred).item()
            else:
                pred_count = torch.sum(pred, dim=(0,1,3,4)).cpu().numpy()

        if len(img.shape) == 4:
            gt_count = gt.shape[1]
        else:
            gt_count = np.array([pts.shape[0] for i, pts in enumerate(gt[0]) if i%4==3])

        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2

        if len(img.shape) == 4:
            self.log_dict({'val/MSE': mse, 'val/MAE': mae})
        else:
            self.log_dict({'val/MSE': np.average(mse), 'val/MAE': np.average(mae)})

    def test_step(self, batch, batch_idx):
        img, gt = batch
        if len(img.shape) == 4:
            b, _, h, w = img.shape
        else:
            b, _, _, h, w = img.shape
        assert b == 1, 'batch size should be 1 in testing'

        if h >= 512 or w >= 512:
            img_patches = []
            pred_count = 0
            h_stride = int(np.ceil(1.0 * h / 512))
            w_stride = int(np.ceil(1.0 * w / 512))
            h_step = h // h_stride
            w_step = w // w_stride
            for i in range(h_stride):
                for j in range(w_stride):
                    h_start = i * h_step
                    if i != h_stride - 1:
                        h_end = (i + 1) * h_step
                    else:
                        h_end = h
                    w_start = j * w_step
                    if j != w_stride - 1:
                        w_end = (j + 1) * w_step
                    else:
                        w_end = w
                    img_patches.append(img[..., h_start:h_end, w_start:w_end])
            
            for patch in img_patches:
                pred = self.forward(patch)
                pred_count += torch.sum(pred).item()

        else:
            pred = self.forward(img)
            pred_count = torch.sum(pred).item()

        gt_count = gt.shape[1]

        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        
        self.log_dict({'test/MSE': mse, 'test/MAE': mae})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-6, verbose=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr, total_steps=self.trainer.estimated_stepping_batches
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        #return {'optimizer': optimizer}

if __name__ == '__main__':
    cli = LightningCLI(Trainer, save_config_overwrite=True)