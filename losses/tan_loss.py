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

    return loss

def count_loss(pred, gt, lamda):
    return nn.MSELoss()(pred, gt) + lamda * sl_loss(pred, gt)

class TAN_Loss(nn.Module):
    def __init__(self, lamda, beta):
        super(TAN_Loss, self).__init__()
        self.lamda = lamda
        self.beta = beta

    @torch.cuda.amp.autocast()
    def forward(self, pred_maps, pred_counts, gt_maps, gt_counts):
        seq_len = pred_maps.shape[1]
        loss = torch.zeros(seq_len)
        for i in range(seq_len):
            loss[i] += sl_loss(pred_maps[:, i, ...], gt_maps[:, i, ...])
            loss[i] += count_loss(pred_counts[:, i], gt_counts[:, i], self.lamda)
        loss /= seq_len
        return loss.mean()

if __name__ == '__main__':
    m = TAN_Loss(0.5, 0.5)
    x = torch.randn(2, 5, 3, 256, 256)
    y = torch.randn(2, 5, 3, 256, 256)
    z = torch.randn(2, 5)
    w = torch.randn(2, 5)
    r = m(x, z, y, w)
    print(r)