import torch.nn as nn
import torch

from models.coordtok.encoder_modules import (ViT3D,
                                             TransformerDecoder,
                                             PatchEmbed2D)


class Encoder(nn.Module):
    def __init__(self,
                 embed_dim=1024,
                 num_layers=24,
                 num_heads=16,
                 patch_type='transformer',
                 patch_num_layers=8,
                 patch_size_xy=16,
                 patch_size_t=8,
                 latent_resolution_xy=16,
                 latent_resolution_t=32,
                 latent_patch_size_xy=8,
                 latent_patch_size_t=16,
                 latent_n_features=8,
                 img_size=128,
                 num_frames=128):
        super().__init__()
        if patch_type == 'transformer':
            self.encoder = ViT3D(img_size=img_size,
                                             frames=num_frames,
                                             patch_size_xy=patch_size_xy,
                                             patch_size_t=patch_size_t,
                                             embed_dim=embed_dim,
                                             num_layers=patch_num_layers,
                                             num_heads=num_heads)
        else:
            raise NotImplementedError # We may use TimeSFormer or ViViT
        self.cross_pos_emb = nn.Parameter(torch.randn(1,
                                                      self.encoder.patch_embed.num_patches,
                                                      embed_dim)) # naive imp now.
        self.latent_resolution_xy = latent_resolution_xy
        self.latent_resolution_t  = latent_resolution_t
        self.latent_patch_size_xy = latent_patch_size_xy
        self.latent_patch_size_t  = latent_patch_size_t

        channel = latent_n_features
        self.channel = channel

        self.patch_embeds_xy = PatchEmbed2D((latent_resolution_xy, latent_resolution_xy),
                                            (latent_patch_size_xy, latent_patch_size_xy),
                                            in_chans=channel, embed_dim=embed_dim)
        self.patch_embeds_yt = PatchEmbed2D((latent_resolution_t, latent_resolution_xy),
                                            (latent_patch_size_t, latent_patch_size_xy),
                                             in_chans=channel, embed_dim=embed_dim)
        self.patch_embeds_xt = PatchEmbed2D((latent_resolution_t, latent_resolution_xy),
                                            (latent_patch_size_t, latent_patch_size_xy),
                                            in_chans=channel, embed_dim=embed_dim)

        self.transformer = TransformerDecoder(num_layers=num_layers,
                                                          num_heads=num_heads,
                                                          inner_dim=embed_dim,
                                                          cond_dim=embed_dim)
        self.projectors_xy = nn.Linear(embed_dim, latent_patch_size_xy*latent_patch_size_xy*channel)
        self.projectors_yt = nn.Linear(embed_dim, latent_patch_size_t*latent_patch_size_xy*channel)
        self.projectors_xt = nn.Linear(embed_dim, latent_patch_size_t*latent_patch_size_xy*channel)

        self.patch_size_t = patch_size_t

    def unpatchify(self, x, is_xy=True):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        h = self.latent_resolution_xy // self.latent_patch_size_xy if is_xy else self.latent_resolution_t // self.latent_patch_size_t
        w = self.latent_resolution_xy // self.latent_patch_size_xy
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        p1 = self.latent_patch_size_xy if is_xy else self.latent_patch_size_t
        p2 = self.latent_patch_size_xy

        x = x.reshape(x.shape[0], h, w, p1, p2, self.channel).permute(0,1,3,2,4,5)
        x = x.reshape(x.shape[0], h*p1, w*p2, self.channel)
        return x

    def forward(self, imgs, weight_tokens, n_frames=None):
        b, f = imgs.shape[:2]
        f = f // self.patch_size_t
        n_frames = n_frames // self.patch_size_t

        imgs = imgs.permute(0, 4, 1, 2, 3)
        image_feats, memory_mask = self.encoder(imgs, n_frames=n_frames)
        image_feats = image_feats + self.cross_pos_emb.to(image_feats.dtype)

        weight_tokens_xy, weight_tokens_yt, weight_tokens_xt = weight_tokens
        weight_tokens_xy = weight_tokens_xy.permute(0, 3, 1, 2)
        weight_tokens_yt = weight_tokens_yt.permute(0, 3, 1, 2)
        weight_tokens_xt = weight_tokens_xt.permute(0, 3, 1, 2)

        weight_tokens = torch.cat([self.patch_embeds_xy(weight_tokens_xy),
                                   self.patch_embeds_yt(weight_tokens_yt),
                                   self.patch_embeds_xt(weight_tokens_xt)], dim=1)
        weight_tokens = self.transformer(weight_tokens, image_feats,
                                         memory_key_padding_mask=memory_mask)#, memory_is_causal=True)

        xy_weights = self.projectors_xy(weight_tokens[:, :self.patch_embeds_xy.num_patches])
        yt_weights = self.projectors_yt(weight_tokens[:, self.patch_embeds_xy.num_patches:self.patch_embeds_xy.num_patches+self.patch_embeds_xt.num_patches])
        xt_weights = self.projectors_xt(weight_tokens[:, self.patch_embeds_xy.num_patches+self.patch_embeds_xt.num_patches:])

        xy_weights = self.unpatchify(xy_weights, is_xy=True)
        yt_weights = self.unpatchify(yt_weights, is_xy=False)
        xt_weights = self.unpatchify(xt_weights, is_xy=False)
        return xy_weights, yt_weights, xt_weights

