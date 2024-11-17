import os
import glob
import random

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms as TF
import torchvision.transforms.functional as F

from decord import VideoReader, cpu


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of 0 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


class VideoTimeSplit(Dataset):
    def __init__(self,
                 path_to_video,
                 num_frames=128,
                 video_type='avi',
                 img_size=128):
        super().__init__()
        '''
            [path_to_video]
                - folder 1
                    - video_name1.[video_type]
                    - video_name2.[video_type]
                - folder n
        '''
        self.img_size = img_size
        self.num_frames = num_frames
        self.vid_files = []
        for idx, path in enumerate(sorted(os.listdir(path_to_video))):
            path_ = os.path.join(path_to_video, path)
            self.vid_files += sorted(glob.glob(f'{path_}/*.{video_type}'))
        self.encode_transforms = TF.Compose([TF.Resize((img_size), interpolation=TF.InterpolationMode.BICUBIC),
                                             TF.CenterCrop(img_size),
                                             TF.ToTensor()])

    def __len__(self):
        return len(self.vid_files)

    def __getitem__(self, idx, fixed_sampling=False):
        vr = VideoReader(self.vid_files[idx], num_threads=1, ctx=cpu(0))

        if fixed_sampling: #For inference
            begin_idx = 0
            if len(vr) >= self.num_frames:
                length = self.num_frames
            else:
                length = len(vr)
        elif len(vr) >= self.num_frames:
            begin_idx = random.randint(0, len(vr) - self.num_frames)
            length = self.num_frames
        else:
            begin_idx = 0
            length = len(vr)

        frames = vr.get_batch(range(begin_idx, begin_idx + length)).asnumpy()
        encode_frames = []
        for frame in frames:
            frame = F.to_pil_image(frame)
            frame = self.encode_transforms(frame)
            encode_frames.append(frame)
        for _ in range(self.num_frames - length):
            frame_empty = -1 * torch.ones_like(frame)
            encode_frames.append(frame_empty)
        input_video = torch.stack(encode_frames).permute(0, 2, 3, 1)

        return input_video, length


##TODO: FIXME
class VideoTimeSplitInference(VideoTimeSplit):
    def __init__(self,
                 path_to_video,
                 num_frames=50,
                 num_videos=-1,
                 video_type='avi',
                 img_size=64):
        super().__init__(path_to_video, num_frames, num_videos, video_type, img_sizep)
        with open(os.path.join(path_to_video, 'evallist.txt'), 'r') as f:
            eval_files = f.readlines()
        eval_files = [eval_file.split() for eval_file in eval_files]

        if '128' in path_to_video:
            self.eval_indices = [(int(eval_file[1]), int(eval_file[2])) for eval_file in eval_files]
            self.fvd_indices  = self.eval_indices
        else:
            if num_frames == 16:
                self.eval_indices = [(int(eval_file[3]), int(eval_file[4])) for eval_file in eval_files]
                self.fvd_indices  = self.eval_indices
            elif num_frames == 128:
                self.eval_indices = [(int(eval_file[1]), int(eval_file[2])) for eval_file in eval_files]
                self.fvd_indices = [(int(eval_file[3]), int(eval_file[4])) for eval_file in eval_files]
            else:
                raise NotImplementedError ##TODO: #frame = 17

        self.vid_files = [os.path.join(path_to_video, eval_file[0]) for eval_file in eval_files]

    def __getitem__(self, idx):
        vr = VideoReader(self.vid_files[idx], num_threads=1, ctx=cpu(0))
        begin_idx, end_idx = self.eval_indices[idx]
        if end_idx == -1:
            end_idx = len(vr)
        length = end_idx - begin_idx
        frames = vr.get_batch(range(begin_idx, begin_idx + length)).asnumpy()
        encode_frames = []
        for frame in frames:
            frame = F.to_pil_image(frame)
            frame = self.encode_transforms(frame)
            encode_frames.append(frame)
        for _ in range(self.num_frames - length):
            frame_empty = -1 * torch.ones_like(frame)
            encode_frames.append(frame_empty)

        input_video = torch.stack(encode_frames).permute(0, 2, 3, 1)
        return input_video, length, (begin_idx, self.fvd_indices[idx])


class VideoTimeWrapperPatchMultiGPU(Dataset):
    def __init__(self,
                 datasets,
                 num_views,
                 batch_size=307200,
                 epochs=1000000,
                 n_gpus=1,
                 rank=0,
                 patch_size=(2, 2, 2)):
        self.datasets = datasets
        self.n_gpus = n_gpus
        self.rank = rank
        self.num_chunk = num_views // self.n_gpus
        self.N_samples = batch_size // self.num_chunk
        self.epochs = epochs

        indices_range = len(self.datasets) // self.n_gpus

        self.start_idx = self.rank * indices_range
        if self.rank == self.n_gpus - 1:
            self.end_idx = len(self.datasets)
        else:
            self.end_idx = (self.rank + 1) * indices_range

        self.p_t, self.p_x, self.p_y = patch_size

    def __len__(self):
        return self.epochs * self.num_chunk

    def get_single_item(self, idx):
        encode_video, n_frames = self.datasets.__getitem__(idx)

        sidelength = encode_video.shape[1:-1]
        patch_max = (self.datasets.num_frames // self.p_t, sidelength[0] // self.p_x, sidelength[1] // self.p_y)
        t_starts = torch.arange(0, n_frames - self.p_t + 1, self.p_t)
        x_starts = torch.arange(0, sidelength[0] - self.p_x + 1, self.p_x)
        y_starts = torch.arange(0, sidelength[1] - self.p_y + 1, self.p_y)

        rand_t_idx = torch.randint(0, len(t_starts), (self.N_samples,))
        rand_x_idx = torch.randint(0, len(x_starts), (self.N_samples,))
        rand_y_idx = torch.randint(0, len(y_starts), (self.N_samples,))

        gt = torch.stack([encode_video[t_starts[rand_t_idx[i]]:t_starts[rand_t_idx[i]]+self.p_t,
                                       x_starts[rand_x_idx[i]]:x_starts[rand_x_idx[i]]+self.p_x,
                                       y_starts[rand_y_idx[i]]:y_starts[rand_y_idx[i]]+self.p_y, :]
                              for i in range(self.N_samples)]).reshape(self.N_samples, -1) # [4096, p_t, p_x, p_y, 3]
        patch_idx_t = t_starts[rand_t_idx] // self.p_t / (patch_max[0]-1)
        patch_idx_x = x_starts[rand_x_idx] // self.p_x / (patch_max[1]-1)
        patch_idx_y = y_starts[rand_y_idx] // self.p_y / (patch_max[2]-1)
        input = torch.stack((patch_idx_t, patch_idx_x, patch_idx_y), dim=1) # [4096, 3]

        return encode_video, input, gt, n_frames, idx

    def __getitem__(self, idx):
        idx = random.sample(range(self.start_idx, self.end_idx), 1)[0]
        return self.get_single_item(idx)


class VideoTimeWrapperFrameMultiGPU(VideoTimeWrapperPatchMultiGPU):
    def __init__(self,
                 datasets,
                 num_views,
                 batch_size=307200,
                 epochs=500000,
                 n_gpus=1,
                 rank=0,
                 patch_size=(1, 8, 8)):
        super().__init__(datasets, num_views, batch_size, epochs, n_gpus, rank, patch_size)
        img_size = datasets.img_size
        batch_size_per_vid = batch_size // num_views * dist.get_world_size()
        assert batch_size_per_vid % (img_size*img_size//patch_size[1]//patch_size[2]) == 0
        assert patch_size[0] == 1 #for simplicity, current.
        self.num_frames_recon = batch_size_per_vid // (img_size*img_size//patch_size[1]//patch_size[2])

    def get_single_item(self, idx):
        encode_video, n_frames = self.datasets.__getitem__(idx)

        sidelength = encode_video.shape[1:-1]
        patch_max = (self.datasets.num_frames // self.p_t, sidelength[0] // self.p_x, sidelength[1] // self.p_y)
        t_starts = torch.arange(0, n_frames - self.p_t + 1, self.p_t)
        x_starts = torch.arange(0, sidelength[0] - self.p_x + 1, self.p_x)
        y_starts = torch.arange(0, sidelength[1] - self.p_y + 1, self.p_y)

        rand_t_idx_frames = torch.randperm(len(t_starts))[:self.num_frames_recon]
        gt = []
        input = []
        for rand_t_idx_frame in rand_t_idx_frames:
            gt_frame = encode_video[rand_t_idx_frame].reshape(sidelength[0]//self.p_x, self.p_x, sidelength[1]//self.p_y, self.p_y, 3)
            gt_frame = gt_frame.permute(0, 2, 1, 3, 4).reshape(-1, self.p_x*self.p_y*3)
            gt.append(gt_frame)

            patch_idx_t_frame = (t_starts[rand_t_idx_frame] // self.p_t / (patch_max[0]-1)).repeat(gt_frame.shape[0])
            patch_idx_x_frame, patch_idx_y_frame = torch.meshgrid(x_starts, y_starts, indexing='ij')
            patch_idx_x_frame = patch_idx_x_frame.flatten() // self.p_x / (patch_max[1]-1)
            patch_idx_y_frame = patch_idx_y_frame.flatten() // self.p_y / (patch_max[2]-1)
            input_frame = torch.stack((patch_idx_t_frame, patch_idx_x_frame, patch_idx_y_frame), dim=1)
            input.append(input_frame)
        gt = torch.cat(gt, dim=0)
        input = torch.cat(input, dim=0)
        return encode_video, input, gt, n_frames, idx


def collate_fn(batch):
    videos, inputs, gts, n_frames_all = [], [], [], []
    for sample in batch:
        video, input, gt, n_frames, _ = sample
        videos.append(video*2-1)
        inputs.append(input)
        gts.append(gt*2-1)
        n_frames_all.append(n_frames)
    videos = torch.stack(videos, dim=0)
    inputs = torch.stack(inputs, dim=0)
    gts = torch.stack(gts, dim=0)
    n_frames_all = torch.tensor(n_frames_all).unsqueeze(1)
    return videos, inputs, gts, n_frames_all

