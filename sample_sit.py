# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import os
import sys
import argparse

import imageio
import numpy as np

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from tools.utils_sit import find_model, parse_transport_args, parse_ode_args, parse_sde_args
from models.diffusion.sit_models import SiT_models
from transport import create_transport, Sampler

import models.coordtok.coordtok_model as coordtok_model


@torch.no_grad()
def decode_video(model, params,
                 img_size, num_frames,
                 patch_pred=(1, 8, 8), max_num_frames=128):
    batch_size = 1
    nframes = num_frames
    p_t, p_x, p_y = patch_pred

    patch_max = (max_num_frames // p_t, img_size // p_x, img_size // p_y)
    t_starts = torch.arange(0, num_frames - p_t + 1, p_t)
    x_starts = torch.arange(0, img_size - p_x + 1, p_x)
    y_starts = torch.arange(0, img_size - p_y + 1, p_y)

    t_grid, x_grid, y_grid = torch.meshgrid(t_starts, x_starts, y_starts, indexing='ij')
    all_patches_starts = torch.stack([t_grid.flatten(), x_grid.flatten(), y_grid.flatten()], dim=-1)

    output = torch.zeros((1, nframes, img_size, img_size, 3))
    patch_idx_t = all_patches_starts[:, 0] // p_t / (patch_max[0]-1)
    patch_idx_x = all_patches_starts[:, 1] // p_x / (patch_max[1]-1)
    patch_idx_y = all_patches_starts[:, 2] // p_y / (patch_max[2]-1)

    input = torch.stack([patch_idx_t, patch_idx_x, patch_idx_y], dim=1).to(params[0].device)
    input = input.unsqueeze(0).repeat(batch_size, 1, 1)
    pred = model.decode(input, params=params).cpu().reshape(batch_size, -1, p_t, p_x, p_y, 3)
    t_starts, x_starts, y_starts = all_patches_starts[:, 0], all_patches_starts[:, 1], all_patches_starts[:, 2]
    for j in range(pred.shape[1]):
        output[:, t_starts[j]:t_starts[j]+p_t, x_starts[j]:x_starts[j]+p_x, y_starts[j]:y_starts[j]+p_y] = pred[:, j]

    return output


def postprocessing_video(sample, mean_std=None):
    (xy_mean, xy_std), (yt_mean, yt_std), (xt_mean, xt_std) = mean_std
    xy_mean, xy_std = xy_mean.reshape(1, -1, 1, 1).cuda(), xy_std.reshape(1, -1, 1, 1).cuda()
    yt_mean, yt_std = yt_mean.reshape(1, -1, 1, 1).cuda(), yt_std.reshape(1, -1, 1, 1).cuda()
    xt_mean, xt_std = xt_mean.reshape(1, -1, 1, 1).cuda(), xt_std.reshape(1, -1, 1, 1).cuda()

    h_xy, h_yt, h_xt = sample
    h_xy = h_xy * xy_std + xy_mean
    h_yt = h_yt * yt_std + yt_mean
    h_xt = h_xt * xt_std + xt_mean
    h_xy = h_xy.permute(0, 2, 3, 1)
    h_yt = h_yt.permute(0, 2, 3, 1)
    h_xt = h_xt.permute(0, 2, 3, 1)

    return h_xy, h_yt, h_xt


def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    learn_sigma = True

    save_path = os.path.join(args.sit_ckpt_path, 'checkpoints', args.sit_ckpt_iter.zfill(7))
    os.makedirs(save_path, exist_ok=True)

    # load mean and std to normalize triplane representation
    means = []
    stds = []
    with open(os.path.join(args.sit_ckpt_path, 'log.txt'), 'r') as file:
        for i in range(8):
            line = file.readline()
            if 'mean' in line:
                line = line.split('mean')[1]
                numbers = line.split('tensor([')[1].split('])')[0]
                means.append(torch.tensor([float(x) for x in numbers.split(',')]))
            elif 'std' in line:
                line = line.split('std')[1]
                numbers = line.split('tensor([')[1].split('])')[0]
                stds.append(torch.tensor([float(x) for x in numbers.split(',')]))
    xy_mean, yt_mean, xt_mean = means
    xy_std, yt_std, xt_std = stds
    mean_std = (xy_mean, xy_std), (yt_mean, yt_std), (xt_mean, xt_std)

    # Load model
    latent_size = (args.latent_resolution_t, args.latent_resolution_xy)
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=args.latent_n_features,
        num_classes=1, class_dropout_prob=0.0
    ).to(device)
    ckpt_path = os.path.join(args.sit_ckpt_path, 'checkpoints', args.sit_ckpt_iter.zfill(7)+'.pt')
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    sampler = Sampler(transport)
    if mode == "ODE":
        if args.likelihood:
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
            )
        else:
            sample_fn = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol,
                rtol=args.rtol,
                reverse=args.reverse
            )

    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    args.out_features = 3
    first_stage_model = coordtok_model.CoordTok(video_shape=(args.num_frames, args.img_size, args.img_size),
                                                enc_embed_dim=args.enc_embed_dim,
                                                enc_num_layers=args.enc_num_layers,
                                                enc_num_heads=args.enc_num_heads,
                                                enc_patch_size_xy=args.enc_patch_size_xy,
                                                enc_patch_size_t=args.enc_patch_size_t,
                                                enc_patch_type=args.enc_patch_type,
                                                enc_patch_num_layers=args.enc_patch_num_layers,
                                                latent_resolution_xy=args.latent_resolution_xy,
                                                latent_resolution_t=args.latent_resolution_t,
                                                latent_n_features=args.latent_n_features,
                                                latent_patch_size_xy=args.latent_patch_size_xy,
                                                latent_patch_size_t=args.latent_patch_size_t,
                                                dec_embed_dim=args.dec_embed_dim,
                                                dec_num_layers=args.dec_num_layers,
                                                dec_num_heads=args.dec_num_heads,
                                                dec_patch_size_xy=args.dec_patch_size_xy,
                                                dec_patch_size_t=args.dec_patch_size_t,
                                                lpips_loss_scale=0.0).cuda()
    first_stage_model_ckpt = torch.load(args.first_model_ckpt, map_location='cpu')
    ckpt_model = first_stage_model_ckpt['model']
    ckpt_model = {key.replace('module.', ''): value for key, value in ckpt_model.items()}
    message = first_stage_model.load_state_dict(ckpt_model, strict=False)
    first_stage_model.eval()

    vid_save_path = os.path.join(save_path, f'{mode}_{args.sampling_method}_{args.num_sampling_steps}_pred')
    os.makedirs(vid_save_path, exist_ok=True)

    z_xy = torch.randn(1,
                       args.latent_n_features, args.latent_resolution_xy, args.latent_resolution_xy,
                       device=device)
    z_yt = torch.randn(1,
                       args.latent_n_features, args.latent_resolution_t, args.latent_resolution_xy,
                       device=device)
    z_xt = torch.randn(1,
                       args.latent_n_features, args.latent_resolution_t, args.latent_resolution_xy,
                       device=device)
    z = (z_xy, z_yt, z_xt)
    vid_name = f'pred.mp4'
    y = torch.tensor([0] * 1, device=device)
    model_kwargs = dict(y=y.to(torch.long))
    samples = sample_fn(z, model.forward, **model_kwargs)[-1]
    params = postprocessing_video(samples, mean_std=mean_std)
    output = decode_video(first_stage_model, params,
                          img_size=args.img_size,
                          num_frames=args.num_frames,
                          patch_pred=(args.dec_patch_size_t, args.dec_patch_size_xy, args.dec_patch_size_xy),
                          max_num_frames=args.num_frames)
    pred_vid = output.view(args.num_frames, args.img_size, args.img_size, 3).cpu()
    pred_vid = (pred_vid+1)/2
    pred_vid = torch.clamp(pred_vid, 0, 1)
    pred_vid = (pred_vid * 255.0).type(torch.uint8).numpy()



    with imageio.get_writer(os.path.join(vid_save_path, vid_name), fps=25) as video_writer:
        for i in range(pred_vid.shape[0]):
            video_writer.append_data(pred_vid[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)

    mode = sys.argv[1]

    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"

    parser.add_argument('--first_model_ckpt', type=str, default='', help='the path of pretrained model')

    #Encoder args
    parser.add_argument('--enc_embed_dim', type=int, default=1024)
    parser.add_argument('--enc_num_layers', type=int, default=24)
    parser.add_argument('--enc_num_heads', type=int, default=16)
    parser.add_argument('--enc_patch_size_xy', type=int, default=16) #For image, not for latent
    parser.add_argument('--enc_patch_size_t', type=int, default=8)
    parser.add_argument('--enc_patch_type', type=str, default='transformer')
    parser.add_argument('--enc_patch_num_layers', type=int, default=8)

    #Latent args
    parser.add_argument('--latent_resolution_xy', type=int, default=16)
    parser.add_argument('--latent_resolution_t', type=int, default=32)
    parser.add_argument('--latent_n_features', type=int, default=8)
    parser.add_argument('--latent_patch_size_xy', type=int, default=8)
    parser.add_argument('--latent_patch_size_t', type=int, default=16)

    #Renderer args
    parser.add_argument('--dec_embed_dim', type=int, default=1024)
    parser.add_argument('--dec_num_layers', type=int, default=24)
    parser.add_argument('--dec_num_heads', type=int, default=16)
    parser.add_argument('--dec_patch_size_xy', type=int, default=8)
    parser.add_argument('--dec_patch_size_t', type=int, default=1)

    # Dataset args
    parser.add_argument('--data_root', type=str, default='/data')
    parser.add_argument('--video_name', type=str, default='UCF-101_train')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=128)

    #SiT args
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-L/2")
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sit_ckpt_path", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")
    parser.add_argument("--sit_ckpt_iter", type=str, default=None)

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE

    args = parser.parse_known_args()[0]
    main(mode, args)
