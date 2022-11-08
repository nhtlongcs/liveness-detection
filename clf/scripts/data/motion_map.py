import glob
import json
import multiprocessing
from pathlib import Path
import sys
import cv2
import numpy as np
from tqdm import tqdm
import os.path as osp

n_worker = multiprocessing.cpu_count() // 2
meta_data_path = sys.argv[1]
root = Path(meta_data_path) / 'extracted_frames'
save_bk_dir = Path(meta_data_path) / 'bk_map'
save_mo_dir = Path(meta_data_path) / 'motion_map'

test_track_path = osp.join(meta_data_path,"test_tracks.json")
train_track_path = osp.join(meta_data_path,"train_tracks.json")

assert test_track_path and train_track_path, "Paths to test and train tracks are not provided"

with open(test_track_path) as f:
    tracks_test = json.load(f)
with open(train_track_path) as f:
    tracks_train = json.load(f)

all_tracks = tracks_train
for track in tracks_test:
    all_tracks[track] = tracks_test[track]

save_bk_dir.mkdir(exist_ok=True)
save_mo_dir.mkdir(exist_ok=True)


def get_bk_map(info):
    # weighted bk map?
    path, save_name = info
    img = glob.glob(path + "/img1/*.jpg")
    img.sort()
    interval = min(5, max(1, int(len(img) / 200)))
    img = img[::interval][:10]
    # img = img[::interval][:1000]
    imgs = []
    outpath = save_bk_dir / f"{save_name}.jpg"
    if outpath.exists():
        return
    for name in img:
        imgs.append(cv2.imread(name))
    avg_img = np.mean(np.stack(imgs), 0)
    avg_img = avg_img.astype(np.int64)
    cv2.imwrite(str(outpath), avg_img)


def get_motion_map(info):
    track, track_id = info
    for i in range(len(track["frames"])):
        frame_path = track["frames"][i]
        frame_path = root / frame_path
        assert frame_path.exists(), "Frame path does not exist"
        frame = cv2.imread(str(frame_path))
        box = track["boxes"][i]
        if i == 0:
            example = np.zeros(frame.shape, np.int64)
        if i % 7 == 1:
            example[box[1] : box[1] + box[3], box[0] : box[0] + box[2], :] = frame[
                box[1] : box[1] + box[3], box[0] : box[0] + box[2], :
            ]
    avg_filename = str(
        Path(track["frames"][0]).parent.parent.parent.name  # S01
        + "_"
        + Path(track["frames"][0]).parent.parent.name  # c001
        + ".jpg"
    )
    avg_img = cv2.imread(str(save_bk_dir / avg_filename)).astype(np.int64)
    postions = (
        (example[:, :, 0] == 0) & (example[:, :, 1] == 0) & (example[:, :, 2] == 0)
    )
    example[postions] = avg_img[postions]
    cv2.imwrite(str(save_mo_dir / f"{track_id}.jpg"), example)


def parallel_task(task, files):
    with multiprocessing.Pool(n_worker) as pool:
        for imgs in tqdm(pool.imap_unordered(task, files)):
            pass


def extract_bk_map():
    paths = root.glob("*/*/*")
    files = []
    for path in paths:
        files.append((str(path), path.parent.name + "_" + path.name))

    parallel_task(get_bk_map, files)


def extract_mo_map():
    files = []
    for track in all_tracks:
        files.append((all_tracks[track], track))
    parallel_task(get_motion_map, files)


if __name__ == "__main__":
    extract_bk_map()
    extract_mo_map()
