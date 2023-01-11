import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class PRL(nn.Module):
    def __init__(self, method):
        super(PRL, self).__init__()
        gaussian3 = np.zeros((3, 3), dtype=np.float32)
        gaussian3[1, 1] = 1
        gaussian3 = cv2.GaussianBlur(gaussian3, (3, 3), 1)
        gaussian3 = np.reshape(gaussian3, (1, 1, 3, 3))
        self.gaussian3 = torch.from_numpy(gaussian3)

        gaussian5 = np.zeros((5, 5), dtype=np.float32)
        gaussian5[2, 2] = 1
        gaussian5 = cv2.GaussianBlur(gaussian5, (5, 5), 1)
        gaussian5 = np.reshape(gaussian5, (1, 1, 5, 5))
        self.gaussian5 = torch.from_numpy(gaussian5)

        if method == 'MAE':
            self.cost1 = nn.L1Loss()
            self.cost2 = nn.L1Loss()
            self.cost3 = nn.L1Loss()
        elif method == 'MSE':
            self.cost1 = nn.MSELoss()
            self.cost2 = nn.MSELoss()
            self.cost3 = nn.MSELoss()

    # @torch.cuda.amp.autocast()
    def forward(self, preds, gts):
        # b, t, c, h, w = preds.shape
        # preds = preds.view(b*t, c, h, w)
        # gts = gts.view(b*t, c, h, w)

        loss1 = self.cost1(preds, gts)

        with torch.no_grad():
            preds_gau3 = F.conv2d(preds, self.gaussian3.to(preds.device), padding=1)
            gts_gau3 = F.conv2d(gts, self.gaussian3.to(gts.device), padding=1)
        loss2 = self.cost2(preds_gau3, gts_gau3)

        with torch.no_grad():
            preds_gau5 = F.conv2d(preds, self.gaussian5.to(preds.device), padding=2)
            gts_gau5 = F.conv2d(gts, self.gaussian5.to(gts.device), padding=2)
        loss3 = self.cost3(preds_gau5, gts_gau5)

        loss = loss1 + 15 * loss2 + 3 * loss3

        return loss

if __name__ == '__main__':
    x = torch.randn(2, 8, 1, 256, 256)
    y = torch.randn(2, 8, 1, 256, 256)
    loss = PRL('MAE')
    z = loss(x, y)
    print(z)