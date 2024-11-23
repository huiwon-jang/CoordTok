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
<h3 align="center">[<a href="https://huiwon-jang.github.io/coordtok/">project page</a>]</h3>

<img width="100%" src="https://github.com/user-attachments/assets/ea11f54d-0cbe-4478-ac51-e596a7f924b8"/>


### 1. Environment setup
```bash
conda create -n coordtok python=3.9.19
conda activate coordtok
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install timm einops decord imageio[ffmpeg] opencv-python scikit-image gdown configargparse wandb torchdiffeq
```

### 2. Dataset

#### Dataset download
- We download UCF-101 on `[DATA_ROOT]` (e.g., `/data`)
```bash
cd [DATA_ROOT]
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
unrar x UCF101.rar
```

#### Dataset pre-processing
- We use train set of UCF-101 for training CoordTok
```bash
cd CoordTok/data
python split_ucf.py --data_root [DATAROOT] --data_name UCF-101
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

### 3. Training scripts on UCF-101
- We provide training scripts of CoordTok on UCF-101.

#### 1. Tokenization: CoordTok - step 1
- We have N gpus (e.g., `N=8`)
- We need M gradent accumulation (e.g., `M=1` for A100x8)
```bash
torchrun --nnodes=1 --nproc_per_node=N train_coordtok.py \
    --data_root [DATAROOT] \
    --num_views 256 \
    --num_iters 1000001 --accum_iter M \
    --enc_embed_dim 1024 --enc_num_layers 24 --enc_num_heads 16 --enc_patch_num_layers 8 \
    --dec_embed_dim 1024 --dec_num_layers 24 --dec_num_heads 16 \
    --point_per_vid 1024 \
    --allow_tf32 \
    --lpips_loss_scale 0.0
```

#### 2. Tokenization: CoordTok - step 2
- For `[CKPT]`, you must include "xx.ckpt"
```bash
torchrun --nnodes=1 --nproc_per_node=N train_coordtok.py \
    --data_root [DATAROOT] \
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
- For `[CKPT]`, you must include "xx.ckpt"
```bash
torchrun --nnodes=1 --nproc_per_node=N train_sit.py \
    --data_root [DATAROOT] \
    --enc_embed_dim 1024 --enc_num_layers 24 --enc_num_heads 16 --enc_patch_num_layers 8 \
    --dec_embed_dim 1024 --dec_num_layers 24 --dec_num_heads 16 \
    --first_model_ckpt [CKPT] \
    --model SiT-L/2 \
    --global_batch_size 64
```

### 4. Evaluation scripts on UCF-101



### TODOs
* [  ] Upload checkpoints trained on Kinetics600 + UCF-101
