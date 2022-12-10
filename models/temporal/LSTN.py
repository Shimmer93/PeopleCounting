import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from fastai.layers import TimeDistributed

class LSTN(nn.Module):
    def __init__(self, input_size=(360, 640), h_blocks=1, w_blocks=2):
        super(LSTN, self).__init__()
        self.h_blocks = h_blocks
        self.w_blocks = w_blocks
        h_size = int(input_size[0]/4/self.h_blocks)
        w_size = int(input_size[1]/4/self.w_blocks)

        self.vgg16 = VGG16()
        self.stn = STN((h_size, w_size))
    
    def forward(self, x):
        """
        :param x: frame t with size (B, 3, 360, 640)
        :return:
                map_t0: density map at time t0 from VGG-16 with size (B, 1, 90, 160)
                map_t1_blocks: a list of density map blocks at time t1 with size (B, 1, 90/H, 160/W) for each block
                               for Mall dataset H=1 W=2, 2 blocks, then (B, 1, 90, 80)
                               for UCSD dataset H=2 W=2, 4 blocks, then (B, 1, 45, 80)
        """
        map_t0 = self.vgg16(x)
        map_t1_blocks = []

        h_chunks = torch.chunk(map_t0, self.h_blocks, dim=2)
        for cnk in h_chunks:
            cnks = torch.chunk(cnk, self.w_blocks, dim=3)
            for c_ in cnks:
                t1_block = self.stn(c_)
                map_t1_blocks.append(t1_block)
        map_t1_blocks = torch.stack(map_t1_blocks, dim=-1)
        return map_t0, map_t1_blocks


class STN(nn.Module):
    def __init__(self, h_w_size=(90, 80)):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # calculate the input size for linear layer
        h = (h_w_size[0] - 7) + 1
        h = int((h-2)/2)+1
        h = (h - 5) + 1
        h = int((h - 2) / 2) + 1

        w = (h_w_size[1] - 7) + 1
        w = int((w-2)/2)+1
        w = (w - 5) + 1
        w = int((w - 2) / 2) + 1

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * h * w, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        """
        :param x: block of output from VGG16 at time t with size (B, 1, 90, 80) or (B, 1, 45, 80)
                 for Mall dataset H=1 W=2, 2 blocks, then (B, 1, 90, 80)
                 for UCSD dataset H=2 W=2, 4 blocks, then (B, 1, 45, 80)
        :return: output size is the same to input size
        """
        xs = self.localization(x)
        xs = xs.view(x.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


class VGG16(nn.Module):
    def __init__(self, load_weights=False, fix_weights=True):
        super(VGG16, self).__init__()
        # Two M layer pulled out from original vgg16.
        # Last three layer in self.cfg are additional to the original vgg16 first 13 layers.
        self.cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 512, 512, 512, 512, 512, 512, 256, 128, 64]
        self.layers = make_layers(self.cfg)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            set_parameter_requires_grad(self.layers[0:22], fix_weights)  # fix weights for the first ten layers
            self.layers[0:16].load_state_dict(mod.features[0:16].state_dict())
            for i in [16, 18, 20]:
                self.layers[i].load_state_dict(mod.features[i+1].state_dict())
                #self.layers[i].weight.copy_(mod.features[i+1].weight)
                #self.layers[i].bias.copy_(mod.features[i+1].bias)

    def forward(self, x):
        """
        :param x: frame t with size (B, 3, 360, 640)
        :return: output size (B, 1, 90, 160)
        """
        x = self.layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class TemporalLSTN(nn.Module):
    def __init__(self, input_h=360, input_w=640, h_blocks=1, w_blocks=2):
        super(TemporalLSTN, self).__init__()
        self.model = TimeDistributed(LSTN((input_h, input_w), h_blocks, w_blocks))

    def forward(self, x):
        """
        :param x: input with size (B, T, 3, 360, 640)
        :return: output with size (B, T, 1, 90, 160)
        """
        x = self.model(x)
        return x

if __name__ == '__main__':
    m = TemporalLSTN(input_h=512, input_w=512,h_blocks=2,w_blocks=2)
    x = torch.randn(2, 8, 3, 512, 512)
    y, y2 = m(x)
    print(y.shape)