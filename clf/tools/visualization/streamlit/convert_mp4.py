from ast import arg
import os 
import os.path as osp
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, help='input path video or dir')
parser.add_argument('-o', '--output', type=str, help='output dir')

if __name__ == '__main__':
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if osp.isdir(args.input):
        for fname in tqdm(os.listdir(args.input)):
            src_path = osp.join(args.input, fname)
            dst_path = osp.join(args.output, fname)
            command = f'ffmpeg -i {src_path} -vcodec libx264 {dst_path}'
            os.system(command)

    else:
        fname = osp.basename(args.input)
        dst_path = osp.join(args.output, fname)
        command = f'ffmpeg -i {args.input} -vcodec libx264 {dst_path}'
        os.system(command)