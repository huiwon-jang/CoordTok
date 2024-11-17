import os

import torch

import gdown
from tqdm import tqdm

def download(id, fname, root=os.path.expanduser('~/.cache/video-diffusion')):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    gdown.download(id=id, output=destination, quiet=False)
    return destination

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 8192

    pbar = tqdm(total=0, unit='iB', unit_scale=True)
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()


_I3D_PRETRAINED_ID = '1fBNl3TS0LA5FEhZv5nMGJs2_7qQmvTmh'

def load_i3d_pretrained(device=torch.device('cpu')):
    from evals.fvd.pytorch_i3d import InceptionI3d
    i3d = InceptionI3d(400, in_channels=3).to(device)
    filepath = download(_I3D_PRETRAINED_ID, 'i3d_pretrained_400.pt')
    ckpt = torch.load(filepath, map_location=device)
    i3d.load_state_dict(ckpt)
    i3d.eval()
    return i3d