import torch
from torch import nn

import models.coordtok.coordtok_encoder as encoder
import models.coordtok.coordtok_decoder as decoder


class CoordTok(nn.Module):
    def __init__(self,
                 video_shape=(128, 128, 128), # (T, H, W)
                 enc_embed_dim=1024,
                 enc_num_layers=24,
                 enc_num_heads=16,
                 enc_patch_size_xy=16,
                 enc_patch_size_t=8,
                 enc_patch_type='transformer',
                 enc_patch_num_layers=8,
                 latent_resolution_xy=32,
                 latent_resolution_t=32,
                 latent_n_features=8,
                 latent_patch_size_xy=16,
                 latent_patch_size_t=16,
                 dec_embed_dim=1024,
                 dec_num_layers=24,
                 dec_num_heads=16,
                 dec_patch_size_xy=8,
                 dec_patch_size_t=1,
                 lpips_loss_scale=0.0):
        super().__init__()
        self.weight_tokens_xy = nn.Parameter(torch.randn(1, latent_resolution_xy, latent_resolution_xy, latent_n_features))
        self.weight_tokens_yt = nn.Parameter(torch.randn(1, latent_resolution_t, latent_resolution_xy, latent_n_features))
        self.weight_tokens_xt = nn.Parameter(torch.randn(1, latent_resolution_t, latent_resolution_xy, latent_n_features))

        self.encoder = encoder.Encoder(embed_dim=enc_embed_dim,
                                       num_layers=enc_num_layers,
                                       num_heads=enc_num_heads,
                                       patch_size_xy=enc_patch_size_xy,
                                       patch_size_t=enc_patch_size_t,
                                       patch_type=enc_patch_type,
                                       patch_num_layers=enc_patch_num_layers,
                                       latent_resolution_xy=latent_resolution_xy,
                                       latent_resolution_t=latent_resolution_t,
                                       latent_n_features=latent_n_features,
                                       latent_patch_size_xy=latent_patch_size_xy,
                                       latent_patch_size_t=latent_patch_size_t,
                                       img_size=video_shape[1],
                                       num_frames=video_shape[0])
        self.decoder = decoder.Decoder(embed_dim=dec_embed_dim,
                                       num_layers=dec_num_layers,
                                       num_heads=dec_num_heads,
                                       patch_size_xy=dec_patch_size_xy,
                                       patch_size_t=dec_patch_size_t,
                                       latent_resolution_xy=latent_resolution_xy,
                                       latent_resolution_t=latent_resolution_t,
                                       latent_n_features=latent_n_features,
                                       video_shape=video_shape,
                                       lpips_loss_scale=lpips_loss_scale)

    def forward(self, vid, input_coords, targets, n_frames):
        params = self.encode(vid, n_frames)
        loss = self.decoder(input_coords, params, targets)
        return loss

    def encode(self, vid, n_frames):
        weight_tokens_xy = self.weight_tokens_xy.repeat(vid.shape[0], 1, 1, 1)
        weight_tokens_yt = self.weight_tokens_yt.repeat(vid.shape[0], 1, 1, 1)
        weight_tokens_xt = self.weight_tokens_xt.repeat(vid.shape[0], 1, 1, 1)
        weight_tokens = [weight_tokens_xy, weight_tokens_yt, weight_tokens_xt]
        return self.encoder(vid, weight_tokens, n_frames)

    def decode(self, input_coords, params):
        return self.decoder.sample_tokens(input_coords, params)

