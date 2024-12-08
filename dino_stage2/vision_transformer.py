# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn

from utils import trunc_normal_
from pos_embed import get_2d_sincos_pos_embed

class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
        self,
        in_chans=384,
        embed_dim=384,
        norm_layer=None,
        bias=True,
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        x = self.proj(x)
        x = self.norm(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads,
                                                    dropout=attn_drop,batch_first=True)
        
    def forward(self, x, mask=None):
        B, N, C = x.shape
        value, attn = self.multihead_attn(x, x, x, key_padding_mask=mask)
                                      
        return value, attn

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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




class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask, return_attention=False):
        y, attn = self.attn(self.norm1(x),mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, num_classes=0, embed_dim=384, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def read_params(self):
        params=[]
        names=[]
        for name,param in self.lstm.named_parameters():
            params.append(torch.sum(param.data).detach().cpu().numpy())
            names.append(name)
        return params, names



    # def prepare_tokens(self, x, mlm_mask=None):
    #     B, L, _ = x.shape
    #     if mlm_mask is not None:
    #         mask_token = self.mask_token.expand(B, L, -1)
    #         w = mlm_mask.flatten(1).unsqueeze(-1).type_as(mask_token)
    #         x = x * (1 - w) + mask_token * w
    #     cls_tokens = self.cls_token.expand(B, -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     self.pos_drop(x)
    #     return 

    def forward(self, input):
        x, mask, mlm_mask= input
        x = self.prepare_tokens(x,mlm_mask)
        for blk in self.blocks:
            x = blk(x,mask)
        x = self.norm(x)
        
        return x[:,0],x[:,1:]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



class VITRMIM(VisionTransformer):
    def __init__(self, num_classes=0, embed_dim=384, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,norm_layer=nn.LayerNorm,slide_ngrids=1000,  **kwargs):
        super().__init__(num_classes, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                 drop_path_rate,norm_layer)
        
        self.slide_ngrids = slide_ngrids
        num_patches = slide_ngrids**2
        self.patch_embed = PatchEmbed(self.embed_dim, self.embed_dim)
        self.register_buffer('pos_embed', torch.zeros(1, num_patches + 1, self.embed_dim), persistent=False)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], slide_ngrids, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.decoder=nn.TransformerEncoderLayer(self.embed_dim, nhead=2,
                dim_feedforward=2048, dropout=0.1, activation="relu")
        self.cross_decoder = nn.TransformerEncoderLayer(self.embed_dim, nhead=2,
                dim_feedforward=2048, dropout=0.1, activation="relu")



    def length_adjustment(self, x, target_len):
        current_len = x.size(1)
        if current_len == target_len:
            return x
        elif current_len < target_len:
            return nn.functional.interpolate(x.permute(0, 2, 1), \
                                             size=target_len, mode='linear').permute(0, 2, 1)
        else:
            return x[:, :target_len, :]
        
    def coords_to_pos(self, coords, tile_size: int = 256):
        """
        This function is used to convert the coordinates to the positional indices

        Arguments:
        ----------
        coords: torch.Tensor
            The coordinates of the patches, of shape [N, L, 2]
        output: torch.Tensor
            The positional indices of the patches, of shape [N, L]
        """
        coords_ = torch.floor(coords / tile_size)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        return pos.long() + 1  # add 1 for the cls token

    def forward(self, input):
        x, coords, mask, mlm_mask, y= input
        
        z = self.patch_embed(x)
        pos=self.coords_to_pos(coords)
        z = z + self.pos_embed[:, pos, :].squeeze(0)
        #z = self.prepare_tokens(x, mlm_mask)
        ##prepare cls token 
        B, L, _ = z.shape
        if mlm_mask is not None:
            mask_token = self.mask_token.expand(B, L, -1)
            w = mlm_mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            z = z * w + mask_token * (1 - w)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        z = self.pos_drop(torch.cat((cls_tokens, z), dim=1))

        
        for blk in self.blocks:
            z = blk(z,mask)
        z = self.norm(z)
        
        #mlm reconstruction
        x_rec = self.decoder(z[:,1:])
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        mlm_loss= (loss_recon * (~mlm_mask).unsqueeze(-1)).sum() / ((~mlm_mask).sum() + 1e-5)/loss_recon.size(-1)

        #cross reconstruction
        memory = self.length_adjustment(z, y.size(1))
        output = self.cross_decoder(memory)
        crsc_loss = F.mse_loss(output, y)

        return z[:,0], mlm_loss, crsc_loss
    
    def inference(self, input):
        x, coords, mask= input
        mask = torch.cat([torch.zeros_like(mask[:, :1]), mask], dim=1)
        
        z = self.patch_embed(x)
        pos=self.coords_to_pos(coords)
        z = z + self.pos_embed[:, pos, :].squeeze(0)
        #z = self.prepare_tokens(x, mlm_mask)
        ##prepare cls token 
        B, L, _ = z.shape
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        z = self.pos_drop(torch.cat((cls_tokens, z), dim=1))

        
        for blk in self.blocks:
            z = blk(z,mask)
        z = self.norm(z)

        return z[:,0]
        
        

def vit_tiny(**kwargs):
    model = VisionTransformer(
          depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(**kwargs):
    model = VisionTransformer(
        depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(**kwargs):
    model = VisionTransformer(embed_dim=384, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_mtask(**kwargs):
    model = VITRMIM( depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


