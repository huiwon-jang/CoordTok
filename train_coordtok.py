import os
import json
import configargparse
from functools import partial

import models.coordtok.coordtok_model as coordtok_model
import tools.datasets as datasets
import tools.utils_coordtok as utils
import tools.trainer_coordtok as trainer
from evals.fvd.download import load_i3d_pretrained

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb


parser = configargparse.ArgumentParser()

# Model configs
## Encoder args
parser.add_argument('--enc_embed_dim', type=int, default=1024)
parser.add_argument('--enc_num_layers', type=int, default=24)
parser.add_argument('--enc_num_heads', type=int, default=16)
parser.add_argument('--enc_patch_size_xy', type=int, default=16) #For image, not for latent
parser.add_argument('--enc_patch_size_t', type=int, default=8)
parser.add_argument('--enc_patch_type', type=str, default='transformer')
parser.add_argument('--enc_patch_num_layers', type=int, default=8)

## Latent args
parser.add_argument('--latent_resolution_xy', type=int, default=16)
parser.add_argument('--latent_resolution_t', type=int, default=32)
parser.add_argument('--latent_n_features', type=int, default=8)
parser.add_argument('--latent_patch_size_xy', type=int, default=8)
parser.add_argument('--latent_patch_size_t', type=int, default=16)

## Decoder args
parser.add_argument('--dec_embed_dim', type=int, default=1024)
parser.add_argument('--dec_num_layers', type=int, default=24)
parser.add_argument('--dec_num_heads', type=int, default=16)
parser.add_argument('--dec_patch_size_xy', type=int, default=8)
parser.add_argument('--dec_patch_size_t', type=int, default=1)

# General training args
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
parser.add_argument('--num_iters', type=int, default=1000000, help='Number of iterations to train for.')
parser.add_argument('--steps_til_summary', type=int, default=1000)
parser.add_argument('--steps_til_save', type=int, default=50000)
parser.add_argument('--accum_iter', type=int, default=1)
parser.add_argument('--allow_tf32', action='store_true')

# Dataset args
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--prefetch_factor', type=int, default=2)
parser.add_argument('--data_root', type=str, default='/data')
parser.add_argument('--video_name', type=str, default='UCF-101_train')
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--num_frames', type=int, default=128)
parser.add_argument('--num_views', type=int, default=256)
parser.add_argument('--point_per_vid', type=int, default=1024)

# First step
parser.add_argument('--save_dir', type=str, default='./logs')
parser.add_argument('--exp_name', type=str, default='coordtok_large_step1_1M')

# Second step
parser.add_argument('--is_second_step', action='store_true')
parser.add_argument('--first_step_ckpt', type=str, default=None)
parser.add_argument('--lpips_loss_scale', type=float, default=1.0)

# DDP
parser.add_argument('--world_size', default=1, type=int)

parser.add_argument('--resume', action='store_true')

def main():
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", init_method='env://',
                            world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ["RANK"]))
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = 0 * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.is_second_step:
        args.exp_name += '_second'
    args.logging_root = os.path.join(args.save_dir, args.exp_name)
    if rank == 0:
        wandb.init(project='coordtok',
                   name=f"{args.exp_name}",
                   config=vars(args),
                   mode="online")

    args.batch_size = args.point_per_vid * args.num_views // args.accum_iter
    vid_dataset = datasets.VideoTimeSplit(path_to_video=os.path.join(args.data_root, args.video_name),
                                          num_frames=args.num_frames,
                                          video_type='avi',
                                          img_size=args.img_size)
    if not args.is_second_step:
        coord_dataset = datasets.VideoTimeWrapperPatchMultiGPU(vid_dataset,
                                                               num_views=args.num_views // args.accum_iter,
                                                               batch_size=args.batch_size // dist.get_world_size(),
                                                               epochs=args.num_iters * args.accum_iter,
                                                               n_gpus=dist.get_world_size(),
                                                               rank=rank,
                                                               patch_size=(args.dec_patch_size_t, args.dec_patch_size_xy, args.dec_patch_size_xy))
    else:
        coord_dataset = datasets.VideoTimeWrapperFrameMultiGPU(vid_dataset,
                                                               num_views=args.num_views // args.accum_iter,
                                                               batch_size=args.batch_size // dist.get_world_size(),
                                                               epochs=args.num_iters * args.accum_iter,
                                                               n_gpus=dist.get_world_size(),
                                                               rank=rank,
                                                               patch_size=(args.dec_patch_size_t, args.dec_patch_size_xy, args.dec_patch_size_xy))
    dataloader = DataLoader(coord_dataset,
                            shuffle=True,
                            batch_size=args.num_views // dist.get_world_size() // args.accum_iter,
                            pin_memory=True,
                            num_workers=args.num_workers,
                            prefetch_factor=args.prefetch_factor,
                            collate_fn=datasets.collate_fn)

    model = coordtok_model.CoordTok(video_shape=(args.num_frames, args.img_size, args.img_size),
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
                                    lpips_loss_scale=args.lpips_loss_scale if args.is_second_step else 0) ##FIXME: force to 0 at first step
    model = DDP(model.to(device), device_ids=[rank])

    i3d = load_i3d_pretrained('cuda')

    if rank == 0:
        os.makedirs(args.logging_root, exist_ok=True)
        with open(os.path.join(args.logging_root, "args.json"), 'w') as json_file:
            json.dump(vars(args), json_file, indent=4)
        save_vid_dir = os.path.join(args.logging_root, 'videos')
        os.makedirs(save_vid_dir, exist_ok=True)
    else:
        save_vid_dir = os.path.join(args.logging_root, 'videos')

    summary_fn = partial(utils.coordtok_summary,
                         vid_dataset,
                         i3d,
                         patch_pred=(args.dec_patch_size_t,
                                     args.dec_patch_size_xy,
                                     args.dec_patch_size_xy),
                         save_path=save_vid_dir,
                         img_size=args.img_size,
                         num_frames=args.num_frames,
                         point_per_vid=args.point_per_vid)

    dist.barrier()

    if args.is_second_step:
        ckpt = torch.load(args.first_step_ckpt, map_location="cpu")["model"]
        model.load_state_dict(ckpt, strict=False)

    trainer.train(model=model, train_dataloader=dataloader,
                  rank=rank, device=device,
                  epochs=1, lr=args.lr, accum_iter=args.accum_iter,
                  steps_til_summary=args.steps_til_summary, steps_til_save=args.steps_til_save,
                  model_dir=args.logging_root,
                  summary_fn=summary_fn, resume=args.resume)

if __name__ == '__main__':
    main()