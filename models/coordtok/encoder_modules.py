import torch
import torch.nn as nn
import torch.distributed as dist

from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Block, PatchEmbed


class ConditionMaskBlock(nn.Module):
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(inner_dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond, memory_key_padding_mask):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = x + self.cross_attn(self.norm1(x), cond, cond,
                                need_weights=False,
                                key_padding_mask=memory_key_padding_mask)[0]
        before_sa = self.norm2(x)
        x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers: int, num_heads: int,
                 inner_dim: int, cond_dim: int = None, mod_dim: int = None,
                 use_attn_bias = False,
                 eps: float = 1e-6):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionMaskBlock(inner_dim, cond_dim,
                num_heads=num_heads,
                eps=eps,
                attn_bias=use_attn_bias
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)

    def forward(self, x, cond=None, memory_key_padding_mask=None):
        if isinstance(cond, list):
            for layer, cond_ in zip(self.layers, cond):
                x = layer(x, cond_, memory_key_padding_mask)
        else:
            for layer in self.layers:
                x = layer(x, cond, memory_key_padding_mask)
        x = self.norm(x)
        return x


class PatchEmbed2D(PatchEmbed):
    def __init__(self,
                 resolution=(32, 32),
                 patch_size=(16, 16),
                 in_chans=8,
                 embed_dim=1024):
        super().__init__(resolution, patch_size[0], in_chans, embed_dim)
        self.patch_size = patch_size
        self.img_size = resolution
        self.grid_size = tuple([s // p for s, p in zip(resolution, patch_size)])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)


class PatchEmbed3D(nn.Module):
    def __init__(self,
                 img_size=224,
                 frames=32,
                 patch_size_xy=16,
                 patch_size_t=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size_xy = to_2tuple(patch_size_xy)
        assert img_size[1] % patch_size_xy[1] == 0
        assert img_size[0] % patch_size_xy[0] == 0
        assert frames % patch_size_t == 0

        num_patches = (
            (img_size[1] // patch_size_xy[1])
            * (img_size[0] // patch_size_xy[0])
            * (frames // patch_size_t)
        )

        self.img_size = img_size
        self.patch_size = patch_size_xy

        self.frames = frames
        self.t_patch_size = patch_size_t

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size_xy[0]
        self.t_grid_size = frames // patch_size_t

        kernel_size = [patch_size_t] + list(patch_size_xy)
        stride_size = [patch_size_t] + list(patch_size_xy)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride_size
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.frames
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        return x


class ViT3D(nn.Module):
    def __init__(self,
                 img_size=224,
                 frames=32,
                 patch_size_xy=16,
                 patch_size_t=4,
                 embed_dim=768,
                 num_layers=4,
                 num_heads=16):
        super().__init__()
        self.patch_embed = PatchEmbed3D(img_size=img_size,
                                        frames=frames,
                                        patch_size_xy=patch_size_xy,
                                        patch_size_t=patch_size_t,
                                        embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dim))
        self.layers = nn.ModuleList(Block(embed_dim, num_heads=num_heads) for _ in range(num_layers))

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, n_frames):
        x = self.patch_embed(x)

        indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        memory_mask = (indices >= n_frames).unsqueeze(-1).repeat(1, 1, x.shape[2])
        memory_mask = memory_mask.reshape(x.shape[0], -1).to(x.dtype)

        x = x.reshape(x.shape[0], -1, x.shape[-1]) + self.pos_emb.to(x.dtype)
        cls_tokens = self.cls_token.repeat(x.shape[0], 1, 1).to(x.dtype)
        x = torch.cat([cls_tokens, x], dim=1)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x[:, 1:])

        return x, memory_mask

