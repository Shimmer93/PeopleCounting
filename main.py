import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np

from models import Model
from losses import Loss
from datasets import Dataset
from losses.post_prob import Post_Prob

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
    else:
        raise NotImplementedError

class Trainer(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.model = Model(self.hparams.model_name, **self.hparams.model)
        self.loss = Loss(self.hparams.loss_name, device='cuda', **self.hparams.loss)
        self.train_dataset = Dataset(self.hparams.dataset_name, method='train', split_file=self.hparams.train_split, **self.hparams.dataset)
        self.val_dataset = Dataset(self.hparams.dataset_name, method='val', split_file=self.hparams.val_split, **self.hparams.dataset)
        self.test_dataset = Dataset(self.hparams.dataset_name, method='test', split_file=self.hparams.test_split, **self.hparams.dataset)

        if self.hparams.loss_name == 'Bayesian':
            self.post_prob = Post_Prob(device='cuda', **self.hparams.post_prob)
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
        
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        preds = self.forward(imgs)
        preds = torch.sum(preds, dim=(1,2)).cpu()
        targs = torch.tensor([gt.shape[1] for gt in gts]).cpu()
        for pred, targ in zip(preds, targs):
            mse_loss = torch.nn.MSELoss()(pred, targ)
            mae_loss = torch.nn.L1Loss()(pred, targ)
            self.log_dict({'val/MSE': mse_loss.item(), 'val/MAE': mae_loss.item()})

    def test_step(self, batch, batch_idx):
        imgs, gts = batch
        preds = self.forward(imgs)
        preds = torch.sum(preds, dim=0)
        targs = torch.tensor([gt.shape[1] for gt in gts])
        for pred, targ in zip(preds, targs):
            mse_loss = torch.nn.MSELoss()(pred, targ)
            mae_loss = torch.nn.L1Loss()(pred, targ)
            self.log_dict({'test/MSE': mse_loss, 'test/MAE': mae_loss})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return {'optimizer': optimizer}

if __name__ == '__main__':
    cli = LightningCLI(Trainer, save_config_overwrite=True)