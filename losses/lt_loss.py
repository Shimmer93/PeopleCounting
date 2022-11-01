import torch.nn as nn

class LTLoss(nn.Module):
    def __init__(self, type = 'MSE', lamda=0.1):
        super(LTLoss, self).__init__()
        self.closs = nn.MSELoss() if type == 'MSE' else nn.L1Loss()
        self.rloss = nn.MSELoss()
        self.lamda = lamda

    # pred: B x 3 x W x W
    # target: B x 2 x W x W
    def forward(self, pred, target):
        T0 = pred[:, 0, :, :]
        T1 = pred[:, 1, :, :]
        B = pred[:, 2, :, :]
        GT0 = target[:, 0, :, :]
        GT1 = target[:, 1, :, :]
        pred = T0 * GT0 * T1
        pred_sum = pred.sum(dim=(1, 2))
        gt0_sum = GT0.sum(dim=(1, 2))
        reg_loss = self.rloss(pred_sum, gt0_sum)
        count_loss = self.closs(pred + B, GT1)
        return count_loss + self.lamda * reg_loss