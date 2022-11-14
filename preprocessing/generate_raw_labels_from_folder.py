from pathlib import Path
import pandas as pd
import sys

video_dir = Path(sys.argv[1])
print(video_dir)
video_label_path = Path(sys.argv[2])
video_filenames = list(Path(video_dir).glob("*.mp4"))
video_ids = [int(video_filename.stem) for video_filename in video_filenames]
video_filenames = [video_filename.name for video_filename in video_filenames]
labels = [0] * len(video_filenames)

df = pd.DataFrame({
    'filename': video_filenames,
    'label': labels,
    'video_id': video_ids
})
df.sort_values(by="video_id", inplace=True)
df = df[['filename', 'label']]
df.to_csv(str(video_label_path), index=False)
