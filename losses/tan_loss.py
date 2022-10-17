import torch
import torch.nn as nn

def sl_loss(pred, gt):
    loss = 0
    for i in range(len(pred)):
        l1_loss = nn.L1Loss()(pred[i], gt[i])
        l2_loss = nn.MSELoss()(pred[i], gt[i])
        if l1_loss < 1:
            loss += 0.5 * l2_loss
        else:
            loss += l1_loss - 0.5

def count_loss(pred, gt, lamda):
    return nn.MSELoss()(pred, gt) + lamda * sl_loss(pred, gt)

class TAN_Loss(nn.Module):
    def __init__(self, lamda, beta):
        super(TAN_Loss, self).__init__()
        self.lamda = lamda
        self.beta = beta

    @torch.cuda.amp.autocast()
    def forward(self, pred_maps, pred_counts, gt_maps, gt_counts):
        batch_size = pred_maps.shape[0]
        seq_len = pred_maps.shape[1]
        loss = torch.zeros(batch_size).cuda()
        for i in range(seq_len):
            loss[i] += sl_loss(pred_maps[:, i, ...], gt_maps[:, i, ...], self.lamda)
            loss[i] += count_loss(pred_counts[:, i], gt_counts[:, i])
        loss /= seq_len
        return loss