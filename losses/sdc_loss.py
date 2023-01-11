import torch
import torch.nn as nn
import torch.nn.functional as F

class SDCLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(SDCLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.density_losses = nn.ModuleList([nn.MSELoss(), nn.MSELoss(), nn.MSELoss()])
        self.scale_loss = nn.MSELoss()
    
    def forward(self, densities, scale, gt_densities, gt_scale):
        lower_smaps = torch.clamp(torch.floor(gt_scale).to(torch.long), 4, 6) - 4
        lower_smaps = torch.nn.functional.one_hot(lower_smaps, 3).permute(0, 1, 4, 2, 3).squeeze(1).to(torch.float32)

        den_loss = 0
        for i in range(len(densities)):
            lower_smap = lower_smaps[:, i:i+1, :, :]
            lower_smap = F.interpolate(lower_smap, scale_factor=2**(2-i), mode='nearest')
            den_loss += self.alpha * self.density_losses[i](densities[i], gt_densities[i] * lower_smap)
            # den_loss += self.alpha * self.density_losses[i](densities[i], gt_densities[i])

        scale_loss = self.scale_loss(scale, gt_scale)
        
        return self.alpha * den_loss + self.beta * scale_loss