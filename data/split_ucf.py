import os
import configargparse

parser = configargparse.ArgumentParser()
## spliting UCF-101 (full set) -> UCF-101_train (train set)
parser.add_argument('--data_root', type=str, default='/data')
parser.add_argument('--data_name', type=str, default='UCF101')

args = parser.parse_args()

os.makedirs(os.path.join(args.data_root, 'UCF-101_train'), exist_ok=True)
os.makedirs(os.path.join(args.data_root, 'UCF-101_test'), exist_ok=True)

with open('./trainlist01.txt', 'r') as f:
    paths = f.readlines()

paths = [path.split(' ')[0] for path in paths]

video_list = []
for root, dirs, files in os.walk(os.path.join(args.data_root, args.data_name)):
    for file in files:
        if file.endswith('.avi'):
            video_list.append(os.path.join(root.split('/')[-1], file))

for path in paths:
    os.makedirs(f"{os.path.join(args.data_root, 'UCF-101_train')}/{path.split('/')[0]}", exist_ok=True)
    os.makedirs(f"{os.path.join(args.data_root, 'UCF-101_test')}/{path.split('/')[0]}", exist_ok=True)

for video in video_list:
    if video in paths:
        os.system(f"cp {os.path.join(args.data_root, args.data_name, video)} {os.path.join(args.data_root, 'UCF-101_train', video)}")
    else:
        os.system(f"cp {os.path.join(args.data_root, args.data_name, video)} {os.path.join(args.data_root, 'UCF-101_test', video)}")
