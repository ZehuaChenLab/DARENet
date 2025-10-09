# import torch
# import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import torch.utils.checkpoint as checkpoint
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# from torchvision import models
# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

nonlinearity = partial(F.relu, inplace=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
# from functools import partial
from torchvision import models
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# import math
# from mamba.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
# from einops import rearrange, repeat
# from typing import Optional, Union, Type, List, Tuple, Callable, Dict


class DAM(nn.Module):
    def __init__(self, c_size, c_channel, d_channel, is_first=False):
        super(DAM, self).__init__()
        self.is_first = is_first
        self.GPA1 = nn.AdaptiveAvgPool2d((c_size, c_size))
        self.conv1 = nn.Conv2d(in_channels=c_channel, out_channels=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.sigmoid1 = nn.Sigmoid()

        self.GPA2 = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Conv2d(in_channels=c_channel, out_channels=c_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_channel)
        self.conv3 = nn.Conv2d(in_channels=c_channel, out_channels=c_channel, kernel_size=1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(in_channels=d_channel, out_channels=d_channel * 2, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(d_channel * 2)
        self.conv5 = nn.Conv2d(in_channels=c_channel * 2, out_channels=c_channel, kernel_size=1)

    def forward(self, c_x, d_x):
        x1 = self.GPA1(c_x)
        x1 = self.conv1(x1)
        x1 = nonlinearity(self.bn1(x1))
        x1 = self.sigmoid1(x1)
        x1 = torch.mul(c_x, x1)
        x2 = self.GPA2(c_x)
        x2 = self.conv2(x2)
        x2 = nonlinearity(self.bn2(x2))
        x2 = self.conv3(x2)
        x2 = self.sigmoid2(x2)
        x2 = torch.mul(c_x, x2)
        x2 = c_x + x1 + x2
        if (self.is_first == False):
            d_x = self.conv4(d_x)
            d_x = nonlinearity(self.bn4(d_x))
        x = torch.cat((x2, d_x), dim=1)
        x = self.conv5(x)

        return x

class Model1(nn.Module):
    def __init__(self, channel):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2, bias=False)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5, bias=False)
        self.conv4 = nn.Conv2d(channel, channel, kernel_size=5, dilation=8, padding=16, bias=False)
        self.conv5 = nn.Conv2d(channel * 4, channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel)
        self.bn2 = nn.BatchNorm2d(channel)
        self.bn3 = nn.BatchNorm2d(channel)
        self.bn4 = nn.BatchNorm2d(channel)

    def forward(self, x):
        x1 = self.conv2(x)
        x1 = nonlinearity(self.bn1(x1))

        x2 = self.conv1(self.conv2(x))
        x2 = nonlinearity(self.bn2(x2))

        x3 = self.conv1(self.conv2(self.conv3(x)))
        x3 = nonlinearity(self.bn3(x3))

        x4 = self.conv1(self.conv2(self.conv3(self.conv4(x))))
        x4 = nonlinearity(self.bn4(x4))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv5(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        # print(window_size)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        down = None
        if self.downsample is not None:
            down = self.downsample(x)
        return x, down

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
class SwinASPP(nn.Module):
    def __init__(self, input_size, input_dim, out_dim, cross_attn,
                 depth, num_heads, mlp_ratio, qkv_bias, qk_scale,
                 drop_rate, attn_drop_rate, drop_path_rate,
                 norm_layer, aspp_norm, aspp_activation, start_window_size,
                 aspp_dropout, downsample, use_checkpoint):

        super().__init__()

        self.out_dim = out_dim
        if input_size == 24:
            self.possible_window_sizes = [8, 12, 24]
        else:
            self.possible_window_sizes = [i for i in range(start_window_size, input_size // 2 + 1) if
                                          input_size % i == 0]
            # print(self.possible_window_sizes)
        self.layers = nn.ModuleList()
        # [2, 4, 8, 16, 32]
        for ws in self.possible_window_sizes:
            # depth = 1 if ws == input_size else depth
            layer = BasicLayer(dim=int(input_dim),
                               input_resolution=(input_size, input_size),
                               depth=depth,
                               num_heads=num_heads,
                               window_size=ws,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=drop_path_rate,
                               norm_layer=norm_layer,
                               downsample=downsample,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # if cross_attn == 'CBAM':
        #     self.proj = CBAMBlock(input_dim=len(self.layers) * input_dim,
        #                           reduction=12,
        #                           input_size=input_size,
        #                           out_dim=out_dim)
        # else:
        #     self.proj = nn.Linear(len(self.layers) * input_dim, out_dim)

        self.proj1 = nn.Linear(2 * input_dim, out_dim)
        self.proj2 = nn.Linear(2 * input_dim, out_dim)
        self.proj3 = nn.Linear(2 * input_dim, out_dim)
        self.proj = nn.Linear(2 * input_dim, out_dim)

        # Check if needed
        self.norm = norm_layer(out_dim) if aspp_norm else None
        if aspp_activation == 'relu':
            self.activation = nn.ReLU()
        elif aspp_activation == 'gelu':
            self.activation = nn.GELU()
        elif aspp_activation is None:
            self.activation = None

        self.dropout = nn.Dropout(aspp_dropout)

    def forward(self, x):
        """
        x: input tensor (high level features) with shape (batch_size, input_size, input_size, input_dim)

        returns ...
        """
        B, C, H, W = x.shape
        # x = x.view(B, H * W, C)
        x = x.reshape(B, H * W, C)

        features = []
        for layer in self.layers:
            out, _ = layer(x)
            features.append(out)
        # print(len(features))
        feature_1_2 = torch.cat([features[0], features[1]], dim=-1)
        feature_1_2 = self.proj1(feature_1_2)

        feature_1_2_3 = torch.cat([feature_1_2, features[2]], dim=-1)
        feature_1_2_3 = self.proj2(feature_1_2_3)

        feature_1_2_3_4 = torch.cat([feature_1_2_3, features[3]], dim=-1)
        feature_1_2_3_4 = self.proj3(feature_1_2_3_4)
        features = torch.cat([x, feature_1_2_3_4], dim=-1)

        # features = torch.cat(features, dim=-1)
        features = self.proj(features)

        # Check if needed
        if self.norm is not None:
            features = self.norm(x)
        if self.activation is not None:
            features = self.activation(x)
        features = self.dropout(features)
        # print(features.shape)
        return features.view(B, self.out_dim, H, W)

    # def load_from(self, pretrained_path):
    #     pretrained_path = pretrained_path
    #     if pretrained_path is not None:
    #         print("pretrained_path:{}".format(pretrained_path))
    #         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #         pretrained_dict = torch.load(pretrained_path, map_location=device)
    #         if "model" not in pretrained_dict:
    #             print("---start load pretrained modle by splitting---")
    #             pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
    #             for k in list(pretrained_dict.keys()):
    #                 if "output" in k:
    #                     print("delete key:{}".format(k))
    #                     del pretrained_dict[k]
    #             msg = self.load_state_dict(pretrained_dict, strict=False)
    #             # print(msg)
    #             return
    #         pretrained_dict = pretrained_dict['model']
    #         print("---start load pretrained modle of swin encoder---")

    #         model_dict = self.state_dict()
    #         num_layers = len(self.layers)
    #         num_pretrained_layers = set([int(k[7]) for k, v in pretrained_dict.items() if 'layers' in k])

    #         full_dict = copy.deepcopy(pretrained_dict)

    #         layer_dict = OrderedDict()

    #         for i in range(num_layers):
    #             keys = [item for item in pretrained_dict.keys() if f'layers.{i}' in item]
    #             for key in keys:
    #                 for j in num_pretrained_layers:
    #                     if key in layer_dict: continue
    #                     # new_k = "layers." + str(i) + k[8:]
    #                     pre_k = "layers." + str(j) + key[8:]
    #                     pre_v = pretrained_dict.get(pre_k, None)
    #                     if pre_v is not None:
    #                         layer_dict[key] = copy.deepcopy(pre_v)

    #                     for k in list(layer_dict.keys()):
    #                         if k in model_dict:
    #                             if layer_dict[k].shape != model_dict[k].shape:
    #                                 # print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
    #                                 del layer_dict[k]
    #                         elif k not in model_dict:
    #                             del layer_dict[k]
    #         msg = self.load_state_dict(layer_dict, strict=False)

    #         print(f"ASPP Found Weights: {len(layer_dict)}")
    #     else:
    #         print("none pretrain")

def build_aspp(input_size, input_dim, out_dim, config):
    if config.norm_layer == 'layer':
        norm_layer = nn.LayerNorm

    if config.aspp_name == 'swin':
        return SwinASPP(
            input_size=input_size,
            input_dim=input_dim,
            out_dim=out_dim,
            depth=config.depth,
            cross_attn=config.cross_attn,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            qk_scale=config.qk_scale,
            qkv_bias=config.qkv_bias,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            drop_path_rate=config.drop_path_rate,
            norm_layer=norm_layer,
            aspp_norm=config.aspp_norm,
            aspp_activation=config.aspp_activation,
            start_window_size=config.start_window_size,
            aspp_dropout=config.aspp_dropout,
            downsample=config.downsample,
            use_checkpoint=config.use_checkpoint
        )

class ASPPConfig:
    aspp_name = 'swin'
    load_pretrained = False
    cross_attn = 'None'  # set to None to disable

    depth = 2
    num_heads = 8
    start_window_size = 2  ## This means we have 2, 7, 14 as window sizes so 3 level

    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    drop_path_rate = 0.1

    norm_layer = 'layer'
    aspp_norm = False
    aspp_activation = 'relu'  # set to None in order to deactivate
    aspp_dropout = 0.1

    downsample = None
    use_checkpoint = False

class EncoderConfig:
    encoder_name = 'swin'
    load_pretrained = True

    img_size = 512
    window_size = 7

    patch_size = 2
    in_chans = 3
    embed_dim = 96

    depths = [2, 2, 6]
    num_heads = [3, 6, 12]

    low_level_idx = 0
    high_level_idx = 2
    high_level_after_block = True
    low_level_after_block = True

    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    drop_path_rate = 0.1

    norm_layer = 'layer'
    high_level_norm = False
    low_level_norm = True

    ape = False
    patch_norm = True
    use_checkpoint = False

class DecoderConfig:
    decoder_name = 'swin'
    load_pretrained = True
    extended_load = False

    window_size = EncoderConfig.window_size

    num_classes = 9

    low_level_idx = EncoderConfig.low_level_idx
    high_level_idx = EncoderConfig.high_level_idx

    depth = 2
    last_layer_depth = 6
    num_heads = 3
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    drop_path_rate = 0.1
    norm_layer = 'layer'
    decoder_norm = True

    use_checkpoint = False

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.deconv2 = nn.ConvTranspose2d(in_channels, in_channels // 2, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 2)
        self.block1 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(in_channels // 2))
        self.block2 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, padding=1, bias=False),
                                    nn.BatchNorm2d(in_channels // 2))
        self.block3 = nn.Sequential(nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=5, padding=2, bias=False),
                                    nn.BatchNorm2d(in_channels // 2))
        self.conv3 = nn.Conv2d(in_channels * 2, n_filters, 1, bias=False)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.deconv2(x)
        x = nonlinearity(self.norm2(x))
        x1 = nonlinearity(self.block1(x))
        x2 = nonlinearity(self.block2(x))
        x3 = nonlinearity(self.block3(x))
        x = torch.cat((x, x1, x2, x3), 1)
        x = self.conv3(x)
        x = nonlinearity(self.norm3(x))
        return x


class CARNet(nn.Module):
    def __init__(self, img_size=1024, num_classes=1):
        super(CARNet, self).__init__()
        self.img_size = img_size
        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.model1 = Model1(filters[0])


        self.encoder1 = resnet.layer1

        self.encoder2 = resnet.layer2

        self.encoder3 = resnet.layer3

        self.encoder4 = resnet.layer4

        self.dam1 = DAM(1024 // 4, c_channel=filters[0], d_channel=filters[0], is_first=True)
        self.dam2 = DAM(1024 // 8, c_channel=filters[1], d_channel=filters[0])
        self.dam3 = DAM(1024 // 16, c_channel=filters[2], d_channel=filters[1])
        self.dam4 = DAM(1024 // 32, c_channel=filters[3], d_channel=filters[2])

        self.swin_aspp = build_aspp(32, 512, 512, ASPPConfig)


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

        m_x = self.model1(x)

        encoder = self.encoder1(x)
        a1 = self.dam1(encoder, m_x)

        encoder = self.encoder2(encoder)

        a2 = self.dam2(encoder, a1)

        encoder = self.encoder3(encoder)
        a3 = self.dam3(encoder, a2)

        encoder = self.encoder4(encoder)
        a4 = self.dam4(encoder, a3)

        # Center
        aspp_x = self.swin_aspp(a4)

        aspp_x = aspp_x + a4
        # # Decoder
        d4 = self.decoder4(aspp_x)
        d4 = d4 + a3
        d3 = self.decoder3(d4)
        d3 = d3 + a2
        d2 = self.decoder2(d3)
        d2 = d2 + a1
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

    net = CARNet(img_size=1500)
    summary(net, input_size=(4, 3, 1500, 1500))
    macs, params = get_model_complexity_info(net, (3, 1500, 1500), print_per_layer_stat=False)
    print(macs, params)