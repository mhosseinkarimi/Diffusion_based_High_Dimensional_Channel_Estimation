import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  

class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, 4*time_dim),
            nn.SiLU(),
            nn.Linear(4*time_dim, time_dim),
        )
    def forward(self, t):
        return self.net(t)  


class InputBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act  = nn.SiLU()
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, dropout=0.0, groups=8, zero_init=True):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))

        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act2  = nn.SiLU()
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

        if zero_init:
            nn.init.zeros_(self.conv2.weight)
            nn.init.zeros_(self.conv2.bias)

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.drop(self.act2(self.norm2(h))))
        return self.skip(x) + h


class SelfAttention(nn.Module):
    def __init__(self, channels, heads=4, groups=32, dropout=0.0):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.mha  = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.out  = nn.Linear(channels, channels)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H*W).transpose(1, 2)      
        y, _ = self.mha(h, h, h, need_weights=False)          
        y = self.out(y).transpose(1, 2).view(B, C, H, W)
        return x + y


class CrossAttention(nn.Module):
    def __init__(self, channels, context_dim, heads=4, groups=32, dropout=0.0):
        super().__init__()
        self.norm_q = nn.GroupNorm(min(groups, channels), channels)
        self.proj_k = nn.Linear(context_dim, channels)
        self.proj_v = nn.Linear(context_dim, channels)
        self.mha    = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.out    = nn.Linear(channels, channels)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, cond):  
        B, C, H, W = x.shape
        q = self.norm_q(x).view(B, C, H*W).transpose(1, 2)    
        k = self.proj_k(cond)                                  
        v = self.proj_v(cond)                                
        y, _ = self.mha(q, k, v, need_weights=False)          
        y = self.out(y).transpose(1, 2).view(B, C, H, W)
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, with_attn=False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, time_dim)
        self.attn = SelfAttention(out_ch) if with_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
    def forward(self, x, t_emb):
        h = self.res(x, t_emb)
        h = self.attn(h)
        return h, self.down(h)

class BottleneckBlock(nn.Module):
    def __init__(self, ch, time_dim, with_cross_attention=False, context_dim=None):
        super().__init__()
        self.res = ResBlock(ch, ch, time_dim)
        self.self_attn = SelfAttention(ch)
        self.cross_attn = (CrossAttention(ch, context_dim) if with_cross_attention else nn.Identity())
    def forward(self, x, t_emb, cond=None):
        h = self.res(x, t_emb)
        h = self.self_attn(h)
        h = self.cross_attn(h, cond) if not isinstance(self.cross_attn, nn.Identity) and cond is not None else h
        return h

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, with_attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)       
        self.res  = ResBlock(out_ch + out_ch, out_ch, time_dim)       
        self.attn = SelfAttention(out_ch) if with_attn else nn.Identity()
    def forward(self, x, skip, t_emb):
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)                               
        h = self.res(h, t_emb)
        h = self.attn(h)
        return h

class OutputBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.GroupNorm(min(32, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
    def forward(self, x):
        return self.head(x)


class DiffusionUNet(nn.Module):
    def __init__(self,
                 in_channels=2, out_channels=2,
                 base_channels=64, channel_mults=(1,2,4,8),
                 time_dim=256, num_res_per_level=1,
                 with_cross_attention=False, context_dim=None):
        super().__init__()

        # Shared time embedding
        self.time_embed = TimeEmbedding(time_dim)

        # Input
        self.input = InputBlock(in_channels, base_channels)

        # Encoder
        enc = []
        ch = base_channels
        skips = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            for _ in range(num_res_per_level):
                enc.append(EncoderBlock(ch, out_ch, time_dim, with_attn=(mult>=4)))
                ch = out_ch
            skips.append(ch)
        self.encoders = nn.ModuleList(enc)

        # Bottleneck
        self.bottleneck = BottleneckBlock(ch, time_dim, with_cross_attention, context_dim)

        # Decoder
        dec = []
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_per_level):
                dec.append(DecoderBlock(ch, out_ch, time_dim, with_attn=(mult>=4)))
                ch = out_ch
        self.decoders = nn.ModuleList(dec)

        self.out = OutputBlock(ch, out_channels)

    def forward(self, x, t, cond=None):   
        t_emb = self.time_embed(t)         
        h = self.input(x)

        skips = []
        for enc in self.encoders:
            s, h = enc(h, t_emb)
            skips.append(s)

        h = self.bottleneck(h, t_emb, cond)

        for dec in self.decoders:
            s = skips.pop()
            h = dec(h, s, t_emb)

        return self.out(h)
