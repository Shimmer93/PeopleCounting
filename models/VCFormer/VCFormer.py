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
    def forward(self, x):
        return self.net(x)

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

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

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
        
    # x: B x T x P x D
    def forward(self, x):
        b = x.shape[0]
        x = rearrange(x, 'b t p d -> (b t) p d')
        x = self.atten(x) + x
        x = self.feed(x) + x
        x = rearrange(x, '(b t) p d -> b t p d', b = b)

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

class Decoder(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.atten = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.feed = PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        self.reg = nn.Linear(dim, 1)
    
    def forward(self, x):
        output = []
        for i in range(x.shape[1]):
            y = self.atten(x[:, i])
            y = self.feed(y)
            output.append(self.reg(y))
        output = torch.stack(output, dim=1)

        output = output.squeeze(-1)
        b, t, p = output.shape
        a = int(sqrt(p))
        output = output.reshape(b, t, a, a)
        return output

class VCFormer(nn.Module):
    def __init__(self, *, image_size=512, image_patch_size=16, frames=4, dim=512, depth=6, heads=8, mlp_dim=1024, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

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

        self.transformer = SPTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.decoder = Decoder(dim, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, t, p, _ = x.shape
        x += self.pos_embedding.view(1, 1, p, -1)
        x += self.time_embedding.view(1, t, 1, -1)

        x = self.dropout(x)
        
        x = self.transformer(x)
        x = self.decoder(x).unsqueeze(1)

        return x

if __name__ == '__main__':
    def num_of_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    model = VCFormer(image_size = 512, image_patch_size = 16, frames = 8, dim = 512, depth = 6, heads = 8, mlp_dim = 1024, dropout = 0.1, emb_dropout = 0.1)
    print(f'Number of parameters: {num_of_params(model):.2f}M')
    img = torch.randn(2, 3, 8, 512, 512)
    out = model(img)
    print(out.shape)