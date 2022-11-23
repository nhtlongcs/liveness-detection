from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from . import DATASET_REGISTRY, default_loader


@DATASET_REGISTRY.register()
class ImageFolderFromCSV(Dataset):

    def __init__(self,
                 CSV_PATH,
                 IMG_DIR,
                 num_rows=-1,
                 transform=None,
                 **kwargs):
        self.csv_path = Path(CSV_PATH)
        self.img_dir = Path(IMG_DIR)
        assert self.csv_path.exists(
        ), f"CSV file {self.csv_path} does not exist"
        assert self.img_dir.exists(
        ), f"Image directory {self.img_dir} does not exist"

        self.csv = pd.read_csv(CSV_PATH)
        if num_rows > 0:
            self.csv = self.csv.iloc[:num_rows]
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        filename, video_id, frame_id, label = row["filename"], row[
            "video_id"], row["frame_id"], row["label"]
        img_path = self.img_dir / filename
        assert img_path.exists(), f"Image {img_path} does not exist"
        img = default_loader(img_path)
        if self.transform is not None:
            img = self.transform(
                image=img)['image']  # only works with albumentations

        assert label in [0, 1], f"Label {label} is not 0 or 1"
        return {
            'filename': filename,
            'video_id': video_id,
            'frame_id': frame_id,
            'image': img,
            'label': torch.tensor(label).long(),
        }

    def collate_fn(self, batch):
        batch_dict = {
            "images": torch.stack([x['image'] for x in batch]),
            "labels": torch.stack([x['label'] for x in batch]),
            "filenames": [x['filename'] for x in batch],
            "video_ids": [x['video_id'] for x in batch],
            "frame_ids": [x['frame_id'] for x in batch],
        }

        return batch_dict
