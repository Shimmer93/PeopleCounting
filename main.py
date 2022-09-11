import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI



from models import Model
from losses import Loss
from datasets import Dataset
from losses.post_prob import Post_Prob

class Trainer(pl.LightningModule):
    
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        print(self.hparams)
        self.model = Model(self.hparams.model_name, **self.hparams.model)
        self.loss = Loss(self.hparams.loss_name, **self.hparams.loss)
        self.train_dataset = Dataset(self.hparams.dataset_name, method='train', split_file=self.hparams.train_split, **self.hparams.dataset)
        self.val_dataset = Dataset(self.hparams.dataset_name, method='val', split_file=self.hparams.val_split, **self.hparams.dataset)
        self.test_dataset = Dataset(self.hparams.dataset_name, method='test', split_file=self.hparams.test_split, **self.hparams.dataset)

        if self.hparams.loss_name == 'Bayesian':
            self.post_prob = Post_Prob(**self.hparams.post_prob)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
            batch_size=self.hparams.batch_size_train, 
            num_workers=self.hparams.num_workers,
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
        imgs, gts, targs, st_sizes = batch
        preds = self.forward(imgs)
        prob_list = self.post_prob(gts, st_sizes)
        loss = self.loss(prob_list, targs, preds)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gts = batch
        preds = self.forward(imgs)
        preds = torch.sum(preds, dim=0)
        targs = torch.tensor([gt.shape[1] for gt in gts])
        for pred, targ in zip(preds, targs):
            mse_loss = torch.nn.MSELoss()(pred, targ)
            mae_loss = torch.nn.L1Loss()(pred, targ)
            self.log_dict({'val/MSE': mse_loss, 'val/MAE': mae_loss})

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
    cli = LightningCLI(Trainer)