import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

from timm.models.vision_transformer import Block

import losses.lpips as lpips


class Transformer(nn.Module):
    def __init__(self,
                 input_dim = 24,
                 video_shape = (128, 256, 256), #(t, x, y)
                 patch_size = (2, 2, 2), #(t, x, y)
                 embed_dim = 768,
                 n_heads = 8,
                 n_layers = 1,
                 ):
        super().__init__()
        self.video_shape = video_shape
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(input_dim, embed_dim)
        scale = embed_dim ** -0.5
        self.pos_embed = nn.Parameter(scale * torch.randn(1,
                                                          video_shape[0] // patch_size[0],
                                                          video_shape[1] // patch_size[1],
                                                          video_shape[2] // patch_size[2],
                                                          embed_dim), requires_grad=True)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(Block(embed_dim, n_heads, qkv_bias=True))
        self.norm = nn.LayerNorm(embed_dim)

        self.input_dim = input_dim
        self.patch_size = patch_size

    def forward(self, x, coords):
        x = self.patch_embed(x)
        size_t = self.video_shape[0] // self.patch_size[0]
        size_x = self.video_shape[1] // self.patch_size[1]
        size_y = self.video_shape[2] // self.patch_size[2]
        if len(coords.shape) == 3:
            coords[:, :, 0] *= (size_t - 1)
            coords[:, :, 1] *= (size_x - 1)
            coords[:, :, 2] *= (size_y - 1)
            coords = coords.long()

            batch_indices = torch.arange(x.shape[0]).unsqueeze(1).expand(x.shape[0], coords.size(1))
            pos_emb = self.pos_embed.expand(x.shape[0], -1, -1, -1, -1)
            pos_emb = pos_emb[batch_indices, coords[:, :, 0], coords[:, :, 1], coords[:, :, 2]]
        else:
            coords[:, 0] *= (size_t - 1)
            coords[:, 1] *= (size_x - 1)
            coords[:, 2] *= (size_y - 1)
            coords = coords.long()

            pos_emb = self.pos_embed[:, coords[:, 0], coords[:, 1], coords[:, 2]]
        x = x + pos_emb.to(x.dtype)
        for block in self.layers:
            x = block(x)
        x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 embed_dim=1024,
                 num_layers=24,
                 num_heads=16,
                 patch_size_xy=8,
                 patch_size_t=1,
                 latent_resolution_xy=16,
                 latent_resolution_t=32,
                 latent_n_features=8,
                 video_shape=(128, 128, 128), # (T, H, W)
                 lpips_loss_scale=0.0):
        super().__init__()
        self.out_features = 3 * patch_size_xy * patch_size_xy * patch_size_t

        self.plane_resolution_xy = latent_resolution_xy
        self.plane_resolution_t = latent_resolution_t
        self.plane_n_features = latent_n_features

        self.transformer = Transformer(input_dim=latent_n_features*3,
                                       video_shape=video_shape,
                                       patch_size=(patch_size_t, patch_size_xy, patch_size_xy),
                                       embed_dim=embed_dim,
                                       n_layers=num_layers,
                                       n_heads=num_heads)
        self.projector = nn.Linear(embed_dim, patch_size_xy*patch_size_xy*patch_size_t*3)

        if lpips_loss_scale > 0:
            self.lpips_loss = lpips.LPIPS().eval()

        self.lpips_loss_scale = lpips_loss_scale
        self.video_shape = video_shape
        self.patch_size_xy = patch_size_xy
        self.patch_size_t = patch_size_t

    def get_embedding(self, coords, grid, is_xy=True): #[bs, n_points, 2], [bs, res, res, n_features]
        # Scale input to [0, resolution-1] range
        if is_xy:
            scaled_x = coords * (self.plane_resolution_xy - 1)
        else:
            scaled_x = coords
            scaled_x[:, :, 0] *= (self.plane_resolution_t - 1)
            scaled_x[:, :, 1] *= (self.plane_resolution_xy - 1)

        # Get the integer part and fractional part
        x0 = torch.floor(scaled_x).long()
        x1 = x0 + 1
        if is_xy:
            x1 = torch.clamp(x1, max=self.plane_resolution_xy - 1)
        else:
            x1 = torch.stack([torch.clamp(x1[:, :, 0], max=self.plane_resolution_t - 1), torch.clamp(x1[:, :, 1], max=self.plane_resolution_xy - 1)], dim=-1)
        frac = scaled_x - x0.float()

        # Bilinear interpolation
        batch_indices = torch.arange(coords.shape[0]).view(-1, 1).expand_as(x0[:, :, 0])

        c00 = grid[batch_indices, x0[:, :, 0], x0[:, :, 1]]
        c01 = grid[batch_indices, x0[:, :, 0], x1[:, :, 1]]
        c10 = grid[batch_indices, x1[:, :, 0], x0[:, :, 1]]
        c11 = grid[batch_indices, x1[:, :, 0], x1[:, :, 1]]

        c0 = c00 * (1 - frac[:, :, 1:2]) + c01 * frac[:, :, 1:2]
        c1 = c10 * (1 - frac[:, :, 1:2]) + c11 * frac[:, :, 1:2]

        c = c0 * (1 - frac[:, :, 0:1]) + c1 * frac[:, :, 0:1]

        return c

    def unpatchify(self, x):
        #We assume that self.patch_size_t = 1 now.
        """
        x: (N, L, patch_size_xy**2 *patch_size_t *3)
        imgs: (N, T, H, W, 3)
        """
        h = self.video_shape[1] // self.patch_size_xy
        w = self.video_shape[2] // self.patch_size_xy
        x = x.reshape(x.shape[0], h, w, self.patch_size_xy, self.patch_size_xy, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * self.patch_size_xy, h * self.patch_size_xy))
        return imgs

    def get_reconstruction(self, all_coords, params=None):
        batch_size, points_per_sample, _ = all_coords.shape
        params = [params[0].reshape(batch_size, self.plane_resolution_xy, self.plane_resolution_xy, self.plane_n_features),
                  params[1].reshape(batch_size, self.plane_resolution_t, self.plane_resolution_xy, self.plane_n_features),
                  params[2].reshape(batch_size, self.plane_resolution_t, self.plane_resolution_xy, self.plane_n_features)]

        xy_coords = all_coords[:, :, [1, 2]]
        yt_coords = all_coords[:, :, [0, 2]]
        xt_coords = all_coords[:, :, [0, 1]]

        spatial_embedding_xy = self.get_embedding(xy_coords, params[0], is_xy=True)
        spatial_embedding_yt = self.get_embedding(yt_coords, params[1], is_xy=False)
        spatial_embedding_xt = self.get_embedding(xt_coords, params[2], is_xy=False)

        embedding = torch.cat((spatial_embedding_xy, spatial_embedding_yt, spatial_embedding_xt), dim=-1)
        z = self.transformer(embedding, all_coords)
        z = self.projector(z)

        return z

    def forward(self, model_input_all, params=None, targets=None):
        all_coords = model_input_all # [bs, 4096, 3]
        batch_size, points_per_sample, _ = all_coords.shape
        z = self.get_reconstruction(all_coords, params)

        loss = ((z - targets)**2).mean(dim=[1, 2]).sum()

        if self.lpips_loss_scale > 0:
            h = self.video_shape[1] // self.patch_size_xy
            w = self.video_shape[2] // self.patch_size_xy

            z_lpips = self.unpatchify(z.reshape(-1, h*w, z.shape[-1]))
            targets_lpips = self.unpatchify(targets.reshape(-1, h*w, targets.shape[-1]))

            if self.lpips_loss_scale > 0:
                lpips_loss = self.lpips_loss(z_lpips, targets_lpips).sum()
                loss += self.lpips_loss_scale * lpips_loss

        return loss

    @torch.no_grad()
    def sample_tokens(self, model_input_all, params=None):
        all_coords = model_input_all # [bs, 4096, 3]
        output = self.get_reconstruction(all_coords, params)

        return output

