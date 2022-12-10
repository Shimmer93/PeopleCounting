import torch
import torchvision.transforms.functional as F
import os
import numpy as np
from datasets.bayesian_dataset import BayesianDataset
from models.baselines.CSRNet import CSRNet, CSRNext
from models.baselines.SASNet import SASNet
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import OrderedDict

data_dir = '/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/jhu'
# save_dir = '/mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/lightning_logs/version_9201/checkpoints/epoch=214_mse=53890.26_mae=95.21.ckpt'
save_dir = '/mnt/home/zpengac/USERDIR/Crowd_counting/PeopleCounting/logs/lightning_logs/jhu_sasnet/checkpoints/epoch=032_mse=4225443.00_mae=1519.99.ckpt'
os.makedirs('test', exist_ok=True)

dataset = BayesianDataset(data_dir, 512, 8, 1, 'test', False, unit_size=16)
# model = CSRNet()
model = SASNet()
model.eval()
state_dict = torch.load(save_dir, 'cpu')['state_dict']
for k in list(state_dict.keys()):
    if k.startswith('model.'):
        state_dict[k[6:]] = state_dict[k]
        del state_dict[k]
model.load_state_dict(state_dict)

def denormalize(img_tensor):
    # denormalize a image tensor
    img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return img_tensor

for i, (input, gt) in tqdm(enumerate(dataset)):
    if input.shape[1] > 1440 or input.shape[2] > 1440:
        continue
    output = model(input.unsqueeze(0))

    new_input = denormalize(input)
    new_output = output.squeeze(0).detach()
    # new_output = F.resize(new_output, input.shape[1:])

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(new_input.permute(1, 2, 0).cpu().numpy())
    ax.set_title(f'GT count: {len(gt)}')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(new_output.squeeze().detach().cpu().numpy())
    ax2.set_title(f'Predicted count: {output[0].sum().item():.2f}')
    plt.savefig(f'test/{i}.png')
    plt.clf()
    plt.close()