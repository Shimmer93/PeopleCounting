import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from math import sqrt

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.init_weights()

    def forward(self, x):
        return self.net(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.init_weights()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

class SpatialBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.atten = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.feed = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        
    # x: B x T x H x W x D
    def forward(self, x):
        b = x.shape[0]
        h = x.shape[2]
        x = rearrange(x, 'b t h w d -> (b t) (h w) d')
        x = self.atten(x) + x
        x = self.feed(x) + x
        x = rearrange(x, '(b t) (h w) d -> b t h w d', b = b, h = h)

        return x

class TemporalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.atten = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.feed = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        
    # x: B x T x P x D
    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b t p d -> (b p) t d')
        x = self.atten(x) + x
        x = self.feed(x) + x
        x = rearrange(x, '(b p) t d -> b t p d', b = b)

        return x

class LocalTemporalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.atten = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.feed = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        
    # x: B x T x H x W x D
    def forward(self, x):
        b, t, h, w, d = x.shape
        nch = h // 4
        ncw = w // 4
        
        x = rearrange(x, 'b t (nch ch) (ncw cw) d -> (b ch cw) (t nch ncw) d', nch=nch, ncw=ncw)
        x = self.atten(x) + x
        x = self.feed(x) + x
        x = rearrange(x, '(b ch cw) (t nch ncw) d -> b t (nch ch) (ncw cw) d', b=b, ch=4, cw=4, nch=nch, ncw=ncw)

        return x

class SPTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialBlock(dim, heads = heads, dim_head = dim_head, mlp_dim=mlp_dim, dropout = dropout),
                TemporalBlock(dim, heads = heads, dim_head = dim_head, mlp_dim=mlp_dim, dropout = dropout)
            ]))

    # x: B x T x P x D
    def forward(self, x):
        for s, t in self.layers:
            x = s(x)
            x = t(x)

        return x

class DilatedConvBlock(nn.Module):
    def __init__(self, dim, hid_dim, dropout=0.):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, hid_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, hid_dim, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(dim, hid_dim, 3, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(hid_dim * 3)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv2d(hid_dim * 3, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    # x: B x D x H x W
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        xo = torch.cat([x1, x2, x3], dim=1)
        xo = self.bn(xo)
        xo = self.final_conv(xo)
        xo = self.dropout(xo)
        xo = x + xo
        return xo

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class LocalSPEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, patch_size, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.conv_dim = 2
        for _ in range(depth):
            self.conv_dim *= 4
            self.layers.append(nn.ModuleList([
                SpatialBlock(dim, heads = heads, dim_head = dim_head, mlp_dim=mlp_dim, dropout = dropout),
                LocalTemporalBlock(dim, heads = heads, dim_head = dim_head, mlp_dim=mlp_dim, dropout = dropout),
                DilatedConvBlock(self.conv_dim, self.conv_dim, dropout=dropout),
            ]))
        self.patch_size = patch_size

    # x: B x T x H x W x D
    def forward(self, x):
        ph, pw = self.patch_size
        b = x.shape[0]
        scale = 1
        for s, t, c in self.layers:
            scale *= 2
            x = s(x)
            x = t(x)
            x = rearrange(x, 'b t h w (ph pw d) -> (b t) d (h ph) (w pw)', ph=ph//scale, pw=pw//scale)
            x = c(x)
            x = rearrange(x, '(b t) d (h ph) (w pw) -> b t h w (ph pw d)', b=b, ph=ph//scale, pw=pw//scale)

        return x

class Decoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, 128, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 64, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 1, 1)
        self.init_weights()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class VCFormer(nn.Module):
    def __init__(self, *, image_size=512, image_patch_size=16, frames=4, dim=512, depth=3, 
                 heads=8, mlp_dim=1024, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        self.ph = patch_height
        self.pw = patch_width
        self.nh = image_height // patch_height
        self.nw = image_width // patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_spatial_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c f (h p1) (w p2) -> b (f h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
            Rearrange('b (f p) d -> b f p d', f = frames, p = num_spatial_patches)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_spatial_patches, dim))
        self.time_embedding = nn.Parameter(torch.randn(1, frames, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = LocalSPEncoder(dim, depth, heads, dim_head, mlp_dim, [patch_height, patch_width], dropout)

        self.decoder = Decoder(dim)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, t, p, _ = x.shape
        x += self.pos_embedding.view(1, 1, p, -1)
        x += self.time_embedding.view(1, t, 1, -1)
        x = self.dropout(x)
        
        x = rearrange(x, 'b t (h w) d -> b t h w d', h=self.nh, w=self.nw)
        x = self.encoder(x)
        new_h, new_w = x.shape[2], x.shape[3]
        x = rearrange(x, 'b t h w (ph pw d) -> (b t) d (h ph) (w pw)', ph=self.nh//new_h, pw=self.nw//new_w)
        x = self.decoder(x)
        x = rearrange(x, '(b t) d h w -> b d t h w', b=b, t=t)

        return x

if __name__ == '__main__':
    def num_of_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    model = VCFormer(image_size = 512, image_patch_size = 16, frames = 8, dim = 512, depth = 3, heads = 8, mlp_dim = 1024, dropout = 0.1, emb_dropout = 0.1)
    print(f'Number of parameters: {num_of_params(model):.2f}M')
    img = torch.randn(2, 3, 8, 512, 512)
    out = model(img)
    print(out.shape)