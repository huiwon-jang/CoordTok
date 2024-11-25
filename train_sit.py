import os
import logging
import argparse
from glob import glob
from time import time
from copy import deepcopy
from collections import OrderedDict

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import tools.datasets as datasets
import models.coordtok.coordtok_model as coordtok_model
from models.diffusion.sit_models import SiT_models
from transport import create_transport
from tools.utils_sit import find_model, parse_transport_args


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def preprocessing_latent(h_xy, h_yt, h_xt, mean_std=None):
    (xy_mean, xy_std), (yt_mean, yt_std), (xt_mean, xt_std) = mean_std
    xy_mean, xy_std = xy_mean.reshape(1, -1, 1, 1), xy_std.reshape(1, -1, 1, 1)
    yt_mean, yt_std = yt_mean.reshape(1, -1, 1, 1), yt_std.reshape(1, -1, 1, 1)
    xt_mean, xt_std = xt_mean.reshape(1, -1, 1, 1), xt_std.reshape(1, -1, 1, 1)
    h_xy = h_xy.permute(0, 3, 1, 2) # [b, c, w, h]
    h_yt = h_yt.permute(0, 3, 1, 2) # [b, c, w, h]
    h_xt = h_xt.permute(0, 3, 1, 2) # [b, c, w, h]

    h_xy = (h_xy - xy_mean) / xy_std
    h_yt = (h_yt - yt_mean) / yt_std
    h_xt = (h_xt - xt_mean) / xt_std

    return h_xy, h_yt, h_xt
    #h = torch.stack([h_xy, h_yt, h_xt], dim=1) # [b, 3, c, w, h]
    #return h


def gather_and_compute_mean_std(xy_weight, yt_weight, xt_weight):
    xy_mean = xy_weight.mean(dim=(0, 2, 3))
    yt_mean = yt_weight.mean(dim=(0, 2, 3))
    xt_mean = xt_weight.mean(dim=(0, 2, 3))

    xy_var = xy_weight.var(dim=(0, 2, 3), unbiased=False)
    yt_var = yt_weight.var(dim=(0, 2, 3), unbiased=False)
    xt_var = xt_weight.var(dim=(0, 2, 3), unbiased=False)

    dist.all_reduce(xy_mean, op=dist.ReduceOp.SUM)
    dist.all_reduce(yt_mean, op=dist.ReduceOp.SUM)
    dist.all_reduce(xt_mean, op=dist.ReduceOp.SUM)

    dist.all_reduce(xy_var, op=dist.ReduceOp.SUM)
    dist.all_reduce(yt_var, op=dist.ReduceOp.SUM)
    dist.all_reduce(xt_var, op=dist.ReduceOp.SUM)

    world_size = dist.get_world_size()
    xy_mean /= world_size
    yt_mean /= world_size
    xt_mean /= world_size

    xy_var /= world_size
    yt_var /= world_size
    xt_var /= world_size

    xy_std = torch.sqrt(xy_var)
    yt_std = torch.sqrt(yt_var)
    xt_std = torch.sqrt(xt_var)

    return (xy_mean, xy_std), (yt_mean, yt_std), (xt_mean, xt_std)


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())

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
                                                lpips_loss_scale=0.0)
    first_stage_model_ckpt = torch.load(args.first_model_ckpt, map_location='cpu')
    ckpt_model = first_stage_model_ckpt['model']
    ckpt_model = {key.replace('module.', ''): value for key, value in ckpt_model.items()}

    first_stage_model.load_state_dict(ckpt_model, strict=False)
    first_stage_model = DDP(first_stage_model.to(device), device_ids=[rank])
    first_stage_model.eval()

    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_dir = f"{args.results_dir}/{args.exp_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    latent_size = (args.latent_resolution_t, args.latent_resolution_xy)
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=args.latent_n_features,
        num_classes=1, class_dropout_prob=0.0
    )

    # Note that parameter initialization is done within the SiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict["model"])
        ema.load_state_dict(state_dict["ema"])
        opt.load_state_dict(state_dict["opt"])
        args = state_dict["args"]

    requires_grad(ema, False)

    model = DDP(model.to(device), device_ids=[rank])
    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )
    logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    dataset = datasets.VideoTimeSplit(path_to_video=os.path.join(args.data_root, args.video_name),
                                      num_frames=args.num_frames,
                                      video_type='avi',
                                      img_size=args.img_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    eval_chunks = [i for i in range(rank, len(dataset), len(dataset) // 2048 * dist.get_world_size())]
    xy_weights = []
    yt_weights = []
    xt_weights = []
    with torch.no_grad():
        for chunk in eval_chunks:
            x, n_frames = dataset.__getitem__(chunk)
            x = (x*2-1).to(device).unsqueeze(0)
            n_frames = torch.tensor([n_frames]).to(device).reshape(1, 1)
            xy_weight, yt_weight, xt_weight = first_stage_model.module.encode(x, n_frames)
            xy_weights.append(xy_weight.permute(0, 3, 1, 2))
            yt_weights.append(yt_weight.permute(0, 3, 1, 2))
            xt_weights.append(xt_weight.permute(0, 3, 1, 2))
    xy_weights = torch.cat(xy_weights, dim=0)
    yt_weights = torch.cat(yt_weights, dim=0)
    xt_weights = torch.cat(xt_weights, dim=0)
    mean_std = gather_and_compute_mean_std(xy_weights, yt_weights, xt_weights)

    if rank == 0:
        (xy_mean, xy_std), (yt_mean, yt_std), (xt_mean, xt_std) = mean_std
        logger.info(f"xy_mean: {xy_mean.cpu()}")
        logger.info(f"xy_std: {xy_std.cpu()}")
        logger.info(f"yt_mean: {yt_mean.cpu()}")
        logger.info(f"yt_std: {yt_std.cpu()}")
        logger.info(f"xt_mean: {xt_mean.cpu()}")
        logger.info(f"xt_std: {xt_std.cpu()}")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, n_frames in loader:
            x = (x*2-1).to(device)
            y = torch.zeros(x.shape[0], dtype=torch.long).to(device)
            n_frames = n_frames.to(device).unsqueeze(1)
            with torch.no_grad():
                xy_weight, yt_weight, xt_weight = first_stage_model.module.encode(x, n_frames)
                x = preprocessing_latent(xy_weight, yt_weight, xt_weight, mean_std=mean_std)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(model, x, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")

                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save SiT checkpoint:
            if train_steps % args.ckpt_every == 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train SiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_model_ckpt', type=str, default=None)

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

    #Decoder args
    parser.add_argument('--dec_embed_dim', type=int, default=1024)
    parser.add_argument('--dec_num_layers', type=int, default=24)
    parser.add_argument('--dec_num_heads', type=int, default=16)
    parser.add_argument('--dec_patch_size_xy', type=int, default=8)
    parser.add_argument('--dec_patch_size_t', type=int, default=1)

    parser.add_argument('--data_root', type=str, default='/data')
    parser.add_argument('--video_name', type=str, default='UCF-101_train')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=128)

    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument('--exp_name', type=str, default='coordtok_large')

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-L/2")
    parser.add_argument("--epochs", type=int, default=7500)
    parser.add_argument("--global_batch_size", type=int, default=64)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=10_000)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a custom SiT checkpoint")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
