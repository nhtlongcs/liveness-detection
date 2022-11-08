from tqdm import tqdm 
import pandas as pd 
from pathlib import Path 

video_df = pd.read_csv("data/train/labels.csv")
keyframes_df = pd.read_csv("data/train/labels_keyframes.csv")
keyframes_dir = Path("data/train/keyframes")

pbar = tqdm(keyframes_df.iterrows(), total=len(keyframes_df))
for i, row in pbar:
    filename, video_id, frame_id, label = row["filename"], row["video_id"], row["frame_id"], row["label"]
    assert (keyframes_dir/filename).exists(), f"Keyframe {filename} does not exist"
    assert video_df[video_df["filename"] == f"{video_id}.mp4"]["label"].values[0] == label, f"Label mismatch for video {video_id}"