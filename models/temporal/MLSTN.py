import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from fastai.layers import TimeDistributed

class MLSTN(nn.Module):
    def __init__(self, input_size=(72, 112)):
        super(MLSTN, self).__init__()
        self.input_size = input_size
        self.vgg16 = VGG16()
        #self.stn = STN((int(self.input_size[0]//4), int(self.input_size[1]//4)))
        self.stn = STN((input_size[0]//8, input_size[1]//8))
        self.stnt = STN((input_size[0]//2, input_size[1]//2))
    
    @torch.cuda.amp.autocast()
    def forward(self, x, setname='train'):
        """
        :param x: frames t0, t1, t2 with size (B, Frames, C, H, W)=(B, 3, 3, 360, 640)
        :return:
                multi_maps: density maps at time t0, t1, t2 from VGG-16 with size (B, 3, 1, 90, 160)
                map_t3: density maps at time t3 with size (B, 1, 90, 160)
        """
        # each frame is consecutively put into the vgg16 net
        for i in range(x.shape[1]):
            maptemp = self.vgg16(x[:, i, :, :, :])
            if i == 0:
                multi_maps = maptemp.unsqueeze(1)
            else:
                multi_maps = torch.cat((multi_maps, maptemp.unsqueeze(1)), dim=1)

        # concatenate multi maps from t0, t1, t2 by squeezing dim2
        map_t3 = self.stn(multi_maps[:, :, 0, :, :])
        return multi_maps, map_t3

class STN(nn.Module):
    def __init__(self, h_w_size=(90, 80)):
        super(STN, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),     # input channels are modified to 3
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        
        #print(f'hw: {h_w_size}')

        # calculate the input size for linear layer
        h = (h_w_size[0] - 7) + 1
        h = int((h-2)/2)+1
        h = (h - 5) + 1
        h = int((h - 2) / 2) + 1

        w = (h_w_size[1] - 7) + 1
        w = int((w-2)/2)+1
        w = (w - 5) + 1
        w = int((w - 2) / 2) + 1
        
        #print(f'hw_after: {h}, {w}')

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * h * w, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.output_layer = nn.Conv2d(3, 1, kernel_size=1)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        """
        :param x: block of output from VGG16 at time t with size (B, 3, 90, 160)
        :return: output size (B, 1, 90, 160)
        """
        #print(f'x, {x.shape}')
        xs = self.localization(x)
        #print(f'xs, {xs.shape}')
        xs = xs.view(x.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        x = self.output_layer(x)

        return x

class VGG16(nn.Module):
    def __init__(self, load_weights=False, fix_weights=True):
        super(VGG16, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            self.frontend.load_state_dict(mod.features[0:23].state_dict())
    @torch.cuda.amp.autocast()
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        #x = F.upsample(x,scale_factor=2)
        #print(f'vgg_output: {x.shape}')
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

class TemporalMLSTN(nn.Module):
    def __init__(self, input_size=(360, 640)):
        super(TemporalMLSTN, self).__init__()
        self.model = TimeDistributed(MLSTN(input_size))

    @torch.cuda.amp.autocast()
    def forward(self, x):
        """
        :param x: input with size (B, T, 3, 360, 640)
        :return: output with size (B, T, 1, 90, 160)
        """
        x = self.model(x)
        return x

if __name__ == '__main__':
    # test
    model = TemporalMLSTN()
    x = torch.randn(2, 5, 3, 360, 640)
    y = model(x)
    print(y.shape)