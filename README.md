<h1 align="center"> Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction </h1>
<div align="center">
  <a href="https://huiwon-jang.github.io/" target="_blank">Huiwon&nbsp;Jang</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://sihyun.me/" target="_blank">Sihyun&nbsp;Yu</a><sup>1</sup>
  <br>
  <a href="https://alinlab.kaist.ac.kr/shin.html" target="_blank">Jinwoo&nbsp;Shin</a><sup>1</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://people.eecs.berkeley.edu/~pabbeel/" target="_blank">Pieter&nbsp;Abbeel</a><sup>2</sup>&ensp; <b>&middot;</b> &ensp;
  <a href="https://younggyo.me/" target="_blank">Younggyo&nbsp;Seo</a><sup>2</sup>
  <br>
  <sup>1</sup> KAIST &emsp; <sup>2</sup>UC Berkeley &emsp; <br>
</div>
<h3 align="center">[<a href="https://huiwon-jang.github.io/coordtok/">project page</a>] [<a href="https://arxiv.org/abs/2411.14762">arxiv</a>]</h3>

<img width="100%" src="https://github.com/user-attachments/assets/ea11f54d-0cbe-4478-ac51-e596a7f924b8"/>


### 1. Environment setup
```bash
conda create -n coordtok python=3.9.19
conda activate coordtok
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install timm einops decord imageio[ffmpeg] opencv-python scikit-image gdown configargparse wandb torchdiffeq
```

### 2. Checkpoints
We provide checkpoint of CoordTok and CoordTok-SiT-L/2 trained on UCF-101.

### 3. Dataset

#### Dataset download
We download UCF-101 dataset to `[DATA_ROOT]` (e.g., `[DATA_ROOT]=/data`).
```bash
cd [DATA_ROOT]
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
unrar x UCF101.rar
```

#### Dataset pre-processing
We use the training set of UCF-101 to train CoordTok.
```bash
cd CoordTok/data
python split_ucf.py --data_root [DATA_ROOT] --data_name UCF-101
```

#### UCF-101
```
[DATA_ROOT]/UCF-101_train
|-- class1
    |-- video1.avi
    |-- video2.avi
    |-- ...
|-- class2
    |-- video1.avi
    |-- video2.avi
    |-- ...
...
```

### 4. Training scripts on UCF-101
We provide training scripts of CoordTok on UCF-101 in the below.
- CoordTok (1M + 50K iters): [[link](https://drive.google.com/file/d/1beZNVwQeTvLU-23gCrvSCyKbEp4phBgw/view?usp=sharing)]
- CoordTok-SiT-L/2 (600K iters): [[link](https://drive.google.com/file/d/1pT94aD0ujsLKPzIWncXFAWw52Me3uPaw/view?usp=sharing)]

#### 1. Tokenization: CoordTok - step 1
- We have N gpus (e.g., `N=8`).
- We need M gradent accumulation (e.g., `M=1` for A100x8).
```bash
torchrun --nnodes=1 --nproc_per_node=N train_coordtok.py \
    --data_root [DATA_ROOT] \
    --num_views 256 \
    --num_iters 1000001 --accum_iter M \
    --enc_embed_dim 1024 --enc_num_layers 24 --enc_num_heads 16 --enc_patch_num_layers 8 \
    --dec_embed_dim 1024 --dec_num_layers 24 --dec_num_heads 16 \
    --point_per_vid 1024 \
    --allow_tf32 \
    --lpips_loss_scale 0.0
```

#### 2. Tokenization: CoordTok - step 2
- For `[CKPT]`, you must include "xx.ckpt".
```bash
torchrun --nnodes=1 --nproc_per_node=N train_coordtok.py \
    --data_root [DATA_ROOT] \
    --num_views 256 \
    --num_iters 1000001 --accum_iter M \
    --enc_embed_dim 1024 --enc_num_layers 24 --enc_num_heads 16 --enc_patch_num_layers 8 \
    --dec_embed_dim 1024 --dec_num_layers 24 --dec_num_heads 16 \
    --point_per_vid 4096 \
    --allow_tf32 \
    --is_second_step --first_step_ckpt [CKPT] \
    --lpips_loss_scale 1.0
```
#### 3. Video generation: CoordTok-SiT-L/2
- For `[CKPT]`, you must include "xx.ckpt".
```bash
torchrun --nnodes=1 --nproc_per_node=N train_sit.py \
    --data_root [DATA_ROOT] \
    --enc_embed_dim 1024 --enc_num_layers 24 --enc_num_heads 16 --enc_patch_num_layers 8 \
    --dec_embed_dim 1024 --dec_num_layers 24 --dec_num_heads 16 \
    --first_model_ckpt [CKPT] \
    --model SiT-L/2 \
    --global_batch_size 64
```

### 5. Evaluation scripts on UCF-101
#### 1. CoordTok video reconstruction
```python
import torch
from models.coordtok.coordtok_model import CoordTok
from tools.utils_coordtok import decode_video

model = CoordTok(video_shape=(128,128,128), # Shape (T, H, W)
                 enc_embed_dim=1024,
                 enc_num_layers=24,
                 enc_num_heads=16,
                 enc_patch_size_xy=16,
                 enc_patch_size_t=8,
                 enc_patch_type='transformer',
                 enc_patch_num_layers=8,
                 latent_resolution_xy=16,
                 latent_resolution_t=8,
                 latent_n_features=8,
                 latent_patch_size_xy=8,
                 latent_patch_size_t=16,
                 dec_embed_dim=1024,
                 dec_num_layers=24,
                 dec_num_heads=16,
                 dec_patch_size_xy=8,
                 dec_patch_size_t=1,
                 lpips_loss_scale=0).cuda()

x = torch.zeros(1, 128, 128, 128, 3).cuda() # Shape (BS, T, H, W, 3) / Range [-1, 1]
n_frames = torch.tensor([[128]], dtype=torch.int64).cuda() # Shape (BS, 1)

z_xy, z_yt, z_xt = model.encode(x, n_frames) # triplane representation

x_recon = decode_video(model,
                       params=[z_xy, z_yt, z_xt],
                       img_size=128,
                       num_frames=128,
                       patch_pred=(1, 8, 8), # Shape (dec_patch_size_t, dec_patch_size_xy, dec_patch_size_xy)
                       max_num_frames=128,
                       Nslice=1)             # Range [-1, 1]
x_recon = (x_recon+1)/2
x_recon = torch.clamp(x_recon, 0, 1) # Range [0, 1]
```

#### 2. CoordTok-SiT-L/2
We provide script for generating a video from CoorTok-SiT-L/2.
- [CKPT_CoordTok]: Checkpoint of CoordTok including "xx.ckpt".
- [CKPT_SiT_PATH]: Checkpoint of SiT excluding "checkpoints/xx.pt".
- [CKPT_SIT_ITER]: Number of iterations for inference (e.g., 600000).
```bash
python sample_sit.py SDE --sampling-method Euler \
    --data_root [DATA_ROOT] \
    --first_model_ckpt [CKPT_CoordTok] \
    --enc_embed_dim 1024 --enc_num_layers 24 --enc_num_heads 16 --enc_patch_num_layers 8 \
    --dec_embed_dim 1024 --dec_num_layers 24 --dec_num_heads 16 \
    --model SiT-L/2 \
    --num-sampling-steps 250 \
    --sit_ckpt_path [CKPT_SIT_PATH] --sit_ckpt_iter [CKPT_SIT_ITER]
```

### TODOs
* [  ] Upload checkpoints trained on larger dataset

### Note
It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to carry out sanity-check experiments in the near future.