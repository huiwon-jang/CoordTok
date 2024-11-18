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


### 1. Environment setup


### 2. Dataset

#### Dataset download


#### Dataset pre-processing


### 3. Training scripts on UCF-101

#### 1. Tokenization: CoordTok - step 1
- We have N gpus
- We need M gradent accumulation
```bash
torchrun --nnodes=1 --nproc_per_node=N train_coortok.py \
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
- For [CKPT], you must include "xx.ckpt"
```bash
torchrun --nnodes=1 --nproc_per_node=N train_coortok.py \
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
#### 3. Video generation: SiT
- For [CKPT], you must include "xx.ckpt"
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