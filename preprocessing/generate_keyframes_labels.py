import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

keyframe_dir = Path(sys.argv[1])
raw_label_path = Path(sys.argv[2])
out_label_path = Path(sys.argv[3])
raw_labels_df = pd.read_csv(raw_label_path)
raw_labels_df.sort_values(by=["filename"], inplace=True)
new_filenames = []
new_video_ids = []
new_frame_ids = []
new_labels = []
pbar = tqdm(raw_labels_df.iterrows(), total=len(raw_labels_df))
for i, row in pbar:
    video_filename, label = row["filename"], row["label"]
    video_id = video_filename.split(".")[0]
    keyframe_format = f"{video_id}-*.jpg"
    keyframe_belongs_to_video = keyframe_dir.glob(keyframe_format)
    pbar.set_description(f"Processing video {video_id}")
    for keyframe in keyframe_belongs_to_video:
        new_filenames.append(keyframe.name)
        new_video_ids.append(video_id)
        new_frame_ids.append(keyframe.stem.split("-")[1])
        new_labels.append(label)

new_video_ids = list(map(lambda x: int(x), new_video_ids))
new_frame_ids = list(map(lambda x: int(x), new_frame_ids))

new_labels_df = pd.DataFrame({
    "filename": new_filenames,
    "video_id": new_video_ids,
    "frame_id": new_frame_ids,
    "label": new_labels
})
new_labels_df.sort_values(by=["video_id", "frame_id"], inplace=True)
new_labels_df.to_csv(out_label_path, index=False)
print(new_labels_df.head())
