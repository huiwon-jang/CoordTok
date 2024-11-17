import os
import copy

from tqdm.autonotebook import tqdm

import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

import tools.utils_coordtok as utils
from evals.fvd.fvd import frechet_distance

import wandb


def train(model, train_dataloader,
          rank, device,
          epochs, lr, accum_iter,
          steps_til_summary, steps_til_save,
          model_dir,
          summary_fn, resume=False):
    optim = torch.optim.AdamW(lr=lr,
                              params=list(model.parameters()),
                              weight_decay=0.001)
    scaler = GradScaler()
    summaries_dir = os.path.join(model_dir, 'summaries')
    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    if rank == 0:
        utils.cond_mkdir(summaries_dir)
        utils.cond_mkdir(checkpoints_dir)
    dist.barrier()

    total_steps = 0
    tmp_psnr = 0
    best_psnr = 0

    if resume:
        resume_ckpt = torch.load(os.path.join(checkpoints_dir, 'model_best.pth'), map_location='cpu')
        total_steps = resume_ckpt['epoch'] + 1
        model_ckpt = resume_ckpt['model']
        model.load_state_dict(model_ckpt)
        optim.load_state_dict(resume_ckpt['optimizer'])
        if 'best_psnr' in resume_ckpt:
            best_psnr = resume_ckpt['best_psnr']
        if dist.get_rank() == 0:
            print(f'resume epoch: {total_steps}')

    n_gpus = dist.get_world_size()
    eval_chunks = [4+64*(n_gpus*k+rank) for k in range(min((len(train_dataloader.dataset.datasets) - 4) // 64, 32)//n_gpus)]
    with tqdm(total=len(train_dataloader) * epochs // accum_iter) as pbar:
        pbar.update(total_steps)

        for epoch in range(epochs):
            for step, (first_frame_all, model_input_all, gt_all, n_frames_all) in enumerate(train_dataloader):
                model.train()
                vid, n_frames_all, model_input_all, gt_all = first_frame_all.to(device), n_frames_all.to(device), model_input_all.to(device), gt_all.to(device)

                with autocast():
                    train_loss = model(vid, model_input_all, gt_all, n_frames_all)
                scaler.scale(train_loss).backward()
                if (step+1) % accum_iter == 0:
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()
                    dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
                    train_loss = train_loss.item() * accum_iter #approx

                    if total_steps % 100 == 0 and rank == 0:
                        wandb.log({"train_loss": train_loss}, step=total_steps)

                    pbar.set_postfix(loss=f"{train_loss:.4f}")

                    if not total_steps % steps_til_summary:
                        wandb_log_dict = dict(iteration=total_steps)
                        with torch.no_grad():
                            str_w = ""
                            tmp_psnr = 0
                            real_embeddings = []
                            fake_embeddings = []

                            model.eval()
                            for chunk in eval_chunks:
                                psnr, real_emb, fake_emb = summary_fn(model.module, chunk, total_steps, cur_idx=chunk)
                                str_w += f" {round(psnr.item(), 4)}, "
                                tmp_psnr += psnr
                                real_embeddings.append(real_emb[:1]) #use first 16 frames
                                fake_embeddings.append(fake_emb[:1]) #use first 16 frames
                            dist.barrier()

                            tmp_psnr = tmp_psnr.to(device)
                            dist.all_reduce(tmp_psnr, op=dist.ReduceOp.SUM)
                            tmp_psnr = tmp_psnr.item() / dist.get_world_size()
                            tmp_psnr /= len(eval_chunks)

                            real_embeddings = torch.cat(real_embeddings).to(device)
                            fake_embeddings = torch.cat(fake_embeddings).to(device)
                            gathered_real_embeddings = [torch.zeros_like(real_embeddings) for _ in range(dist.get_world_size())]
                            gathered_fake_embeddings = [torch.zeros_like(fake_embeddings) for _ in range(dist.get_world_size())]
                            dist.all_gather(gathered_real_embeddings, real_embeddings)
                            dist.all_gather(gathered_fake_embeddings, fake_embeddings)
                            real_embeddings = torch.cat(gathered_real_embeddings, dim=0)
                            fake_embeddings = torch.cat(gathered_fake_embeddings, dim=0)
                            dist.barrier()
                            if rank == 0:
                                fvd = frechet_distance(fake_embeddings.clone().float().detach(), real_embeddings.clone().float().detach())
                                wandb_log_dict["mean_psnr"] = tmp_psnr
                                wandb_log_dict["fvd"] = fvd
                                if tmp_psnr > best_psnr:
                                    torch.save({'epoch': total_steps,
                                                'model': model.state_dict(),
                                                'optimizer': optim.state_dict(),
                                                'best_psnr': tmp_psnr,
                                                }, os.path.join(checkpoints_dir, 'model_best.pth'))
                                    best_psnr = tmp_psnr
                                tqdm.write(f"Steps %d, Total loss %0.6f, FVD: {fvd:.2f}, PSNR: {str_w}, best PSNR: {best_psnr}" % (total_steps, train_loss))
                                wandb.log(wandb_log_dict, step=total_steps)
                    if not total_steps % steps_til_save:
                        if rank == 0:
                            torch.save({'epoch': total_steps,
                                        'model': model.state_dict(),
                                        'best_psnr': tmp_psnr,
                                        }, os.path.join(checkpoints_dir, f"model_{total_steps//1000}k.pth"))
                    pbar.update(1)
                    total_steps += 1
                    if total_steps > len(train_dataloader):
                        break
        if rank == 0:
            torch.save({'epoch': total_steps,
                        'model': model.state_dict(),
                        'optimizer': optim.state_dict(),
                        }, os.path.join(checkpoints_dir, f'model_final.pth'))

        return

