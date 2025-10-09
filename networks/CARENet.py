import os
import torch.nn.functional as F
from functools import partial
import math
import torch.utils.checkpoint as checkpoint
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import sys
sys.path.append(".")
from MyNet.SnakeMamba import Snake_SS2D

import torch.nn.functional as F

nonlinearity = partial(F.relu, inplace=True)

class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, padding=1, in_chans=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, patch_size, stride=stride, padding=padding),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x):
        return self.proj(x)

class LN(nn.Module):
    def __init__(self, c_channel,norm_layer=nn.LayerNorm):
        super(LN, self).__init__()
        self.ln = nn.LayerNorm(c_channel)

    def forward(self,x):
            x = self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            return x

class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels, kernel_size=scale[i], padding=scale[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MultiScaleDWConv(hidden_features)
        self.act = nn.ReLU()
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.fc2(x)
        return x

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner*2 , bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        return y

class DBSSBlock(nn.Module):
    def __init__(self, dim,dila,
                 mlp_ratio=4):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # self.norm1 = nn.GroupNorm(1, dim)
        self.norm1 = LN(dim)
        self.norm2 = LN(dim)
        self.con1 = nn.Conv2d(dim, dim, kernel_size=3, padding=dila, groups=dim,dilation=dila)

        self.self_attention = Snake_SS2D(d_model=dim//2)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            # nn.Conv2d(dim, dim, kernel_size=3, padding=dila, groups=dim,dilation=dila),
            nn.ReLU(),
            nn.BatchNorm2d(dim),
            ECA(dim)
        )
        # self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        # x = x + self.pos_embed(x)
        shorcut = x.clone()
        x = self.norm1(x)
        x = self.con1(x)
        x = self.self_attention(x)
        x = self.norm2(x)
        x = self.proj(x) + shorcut
        x = self.mlp(self.norm2(x)) + x
        return x

class DBSSM(nn.Module):
    def __init__(self, dim):
        super(DBSSM, self).__init__()
        self.in_proj = nn.Linear(dim, dim * 2)
        blocks = nn.ModuleList()
        blocks.append(DBSSBlock(dim,1))
        blocks.append(DBSSBlock(dim,3))
        blocks.append(DBSSBlock(dim,5))
        blocks.append(DBSSBlock(dim,7))
        # blocks.append(Block1(dim))
        self.convs = blocks
        self.proj = nn.Sequential(
            ECA(5* dim),
            nn.Conv2d(5*dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.ReLU(),
            LN(dim)
        )

    def forward(self, x):
        shorcut = x.clone()
        x = x.permute(0,2,3,1)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0,3,1,2)
        z = z.permute(0, 3, 1, 2)
        res = [x]
        for conv in self.convs:
            x = conv(x)
            res.append(x*F.silu(z))

        res = torch.cat(res, dim=1)
        x = self.proj(res)
        # x = self.con1(x) + shorcut
        return x+shorcut

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPPoolingH(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingH, self).__init__(
            nn.AdaptiveAvgPool2d((32,1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPPPoolingW(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPoolingW, self).__init__(
            nn.AdaptiveAvgPool2d((1,32)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class FEM(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."
        # mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_hidden_dim = dim
        blocks2 = nn.ModuleList()
        blocks3 = nn.ModuleList()
        # self.pos_embed = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.patch = PatchEmbed( in_chans=dim // 2, embed_dim=dim)
        self.norm0 = nn.GroupNorm(1, dim)
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(dim,dim,kernel_size=3,padding=1,groups=dim)
        self.norm1 = nn.GroupNorm(1,dim)
        self.ASPP1 = ASPPPoolingH(in_channels=dim, out_channels=dim)
        self.MaxP1 = nn.MaxPool2d(3,1,padding=1)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

        self.conv1_1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1_1 = nn.GroupNorm(1, dim)
        self.ASPP1_1 = ASPPPoolingW(in_channels=dim, out_channels=dim)
        self.MaxP1_1 = nn.MaxPool2d(3, 1, padding=1)
        self.mlp1_1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.mlp2_1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)
        self.mlp2_1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)


        self.conv2 = nn.Conv2d(dim,dim,(3,1),padding=(1,0))
        self.norm2 = nn.GroupNorm(1, dim)

        self.conv3 = nn.Conv2d(dim,dim,(5,1),padding=(2,0))
        self.norm3 = nn.GroupNorm(1, dim)

        self.conv2_1 = nn.Conv2d(dim,dim,(1,3),padding=(0,1))
        self.norm2_1 = nn.GroupNorm(1, dim)

        self.conv3_1 = nn.Conv2d(dim,dim,(1,5),padding=(0,2))
        self.norm3_1 = nn.GroupNorm(1, dim)

        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            ECA(dim)
        )
        self.norm4 = nn.GroupNorm(1, dim)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim)

    def forward(self, x,y):
        shortcut = x.clone()
        y = self.act(self.norm0(self.patch(y)))
        x = 0.75 * x + 0.25 * y + shortcut

        x1 = self.norm1(self.conv1(x))
        x1_a = self.mlp1(self.ASPP1(x1))
        x1_m = self.mlp2(self.MaxP1(x1))
        x1 = torch.sigmoid(x1_a + x1_m)
        x2 = self.norm2(self.conv2(x))
        x3 = self.norm3(self.conv3(x))

        x1_1 = self.norm1_1(self.conv1_1(x))
        x1_1_a = self.mlp1_1(self.ASPP1_1(x1_1))
        x1_1_m = self.mlp2_1(self.MaxP1_1(x1_1))
        x1_1 = torch.sigmoid(x1_1_a + x1_1_m)
        x2_1 = self.norm2_1(self.conv2_1(x1_1))
        x3_1 = self.norm3_1(self.conv3_1(x))

        x= x2 * x1.expand_as(x2) + x3 *x1.expand_as(x3) + x2_1 * x1_1.expand_as(x2_1) + x3_1 *x1_1.expand_as(x3_1)
        x = self.mlp3(self.norm4(self.proj(x)))
        return x+shortcut

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        # self.norm1 = LN(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        # self.norm2 = LN(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        # self.norm3 = LN(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CARENet(nn.Module):
    def __init__(self, img_size=1024, num_classes=1):
        super(CARENet, self).__init__()
        self.img_size=img_size
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.c2 = FEM(filters[1])
        self.c3 = FEM(filters[2])
        self.c4 = FEM(filters[3])
        self.s4 = DBSSM(filters[3])


        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        if self.img_size != 1024:
            x = F.interpolate(x, size=1024, mode='bilinear', align_corners=True)
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)


        e2 = self.encoder2(e1)
        c2 = self.c2(e2,e1)


        e3 = self.encoder3(e2)
        c3 = self.c3(e3,c2)
        e4 = self.encoder4(e3)
        c4 = self.c4(e4,c3)
        # Center#
        s4 = self.s4(c4)
        # s4 = self.s4(c4)
        # e4 = s4
        e4 = s4
        # Decoder
        d4 = self.decoder4(e4)
        d4 = d4 + c3

        d3 = self.decoder3(d4)
        d3 = d3+c2

        d2 = self.decoder2(d3) 

        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        if self.img_size != 1024:
            out = F.interpolate(out, size=self.img_size, mode='bilinear', align_corners=True)
        return out



if __name__ == '__main__':
    from torchinfo import summary
    from ptflops import get_model_complexity_info

    net = CARENet(img_size=1500)
    summary(net, input_size=(4, 3, 1024, 1024))
    macs, params = get_model_complexity_info(net, (3, 1024, 1024), print_per_layer_stat=False)
    print(macs, params)
