import os

import torch
import torch.distributed as dist

import imageio

from evals.fvd.fvd import get_fvd_logits


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


@torch.no_grad()
def decode_video(model, params,
                 img_size, num_frames,
                 patch_pred=(1, 8, 8), max_num_frames=128, point_per_vid=1024,
                 Nslice=None):
    # num_frames: frame length of the video to be decoded
    # max_num_frames: maximum frame length of the video that coordtok is trained

    nframes = num_frames
    p_t, p_x, p_y = patch_pred
    n_patches = (img_size // p_x) * (img_size // p_y) * num_frames // p_t

    if Nslice == None:
        Nslice = n_patches // point_per_vid

    patch_max = (max_num_frames // p_t, img_size // p_x, img_size // p_y)
    t_starts = torch.arange(0, num_frames - p_t + 1, p_t)
    x_starts = torch.arange(0, img_size - p_x + 1, p_x)
    y_starts = torch.arange(0, img_size - p_y + 1, p_y)

    t_grid, x_grid, y_grid = torch.meshgrid(t_starts, x_starts, y_starts, indexing='ij')
    all_patches_starts = torch.stack([t_grid.flatten(), x_grid.flatten(), y_grid.flatten()], dim=-1)

    split = int(all_patches_starts.shape[0] / Nslice)
    output = torch.zeros((nframes, img_size, img_size, 3))
    for i in range(Nslice):
        split_patches_starts = all_patches_starts[i*split:(i+1)*split]
        patch_idx_t = split_patches_starts[:, 0] // p_t / (patch_max[0]-1)
        patch_idx_x = split_patches_starts[:, 1] // p_x / (patch_max[1]-1)
        patch_idx_y = split_patches_starts[:, 2] // p_y / (patch_max[2]-1)
        input = torch.stack([patch_idx_t, patch_idx_x, patch_idx_y], dim=1).to(params[1].device)
        pred = model.decode(input.reshape(1, -1, 3), params=params).cpu().reshape(-1, p_t, p_x, p_y, 3)

        t_starts, x_starts, y_starts = split_patches_starts[:, 0], split_patches_starts[:, 1], split_patches_starts[:, 2]
        for j in range(pred.shape[0]):
            output[t_starts[j]:t_starts[j]+p_t, x_starts[j]:x_starts[j]+p_x, y_starts[j]:y_starts[j]+p_y] = pred[j]

    return output


@torch.no_grad()
def coordtok_summary(vid_dataset, i3d, model,
                     chunk,
                     total_steps,
                     patch_pred=(1, 8, 8),
                     save_path=None,
                     cur_idx=0,
                     img_size=128,
                     num_frames=128,
                     point_per_vid=1024):
    max_num_frames = num_frames
    gt_vid_encode, num_frames = vid_dataset.__getitem__(chunk, fixed_sampling=True)
    n_frames = torch.tensor(num_frames).reshape(1, 1)
    params = model.encode((gt_vid_encode*2-1).cuda().unsqueeze(0), n_frames.cuda())
    pred_vid = decode_video(model, params,
                            img_size=img_size, num_frames=num_frames,
                            patch_pred=patch_pred,
                            max_num_frames=max_num_frames,
                            point_per_vid=point_per_vid)
    pred_vid = (pred_vid+1)/2
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid_encode = gt_vid_encode[:num_frames]

    psnr = 10*torch.log10(1 / torch.mean((gt_vid_encode - pred_vid)**2))

    real = (gt_vid_encode*255.0).unsqueeze(0).type(torch.uint8).numpy()
    fake = (pred_vid*255.0).unsqueeze(0).type(torch.uint8).numpy()
    real_emb = get_fvd_logits(real, i3d=i3d, device='cuda').cpu()
    fake_emb = get_fvd_logits(fake, i3d=i3d, device='cuda').cpu()

    vid_save_path = os.path.join(save_path, f"{str(total_steps).zfill(6)}")
    if dist.get_rank() == 0:
        os.makedirs(vid_save_path, exist_ok=True)
    dist.barrier()

    save_vid = (pred_vid * 255).byte().numpy()
    video_path = os.path.join(save_path, f"{str(total_steps).zfill(6)}", f"pred_{str(cur_idx).zfill(4)}.mp4")
    with imageio.get_writer(video_path, fps=25) as video_writer:
        for i in range(save_vid.shape[0]):
            video_writer.append_data(save_vid[i])

    return psnr, real_emb, fake_emb

