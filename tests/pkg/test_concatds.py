from core_pkg.dataloader import TwoStreamDataLoader
from core_pkg.dataloader import ConcatDataset

from core.dataset import DATASET_REGISTRY
from core.augmentations import TRANSFORM_REGISTRY
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import pytest


@pytest.mark.order(1)
def test_concat_dataset(dataset_name="ImageFolderFromCSV"):
    image_transform = TRANSFORM_REGISTRY.get('train_classify_tf')(img_size=380)

    uds = DATASET_REGISTRY.get(dataset_name)(
        CSV_PATH="data/train/labels_keyframes_test.csv",
        IMG_DIR="data/train/keyframes",
        transform=image_transform,
        return_lbl=False)
    ds = DATASET_REGISTRY.get(dataset_name)(
        CSV_PATH="data/train/labels_keyframes_test.csv",
        IMG_DIR="data/train/keyframes",
        transform=image_transform,
        return_lbl=True)
    cat_dataset = ConcatDataset([ds, uds])

    for i in range(len(ds) - 5, len(ds) + 5):
        print(cat_dataset[i].keys())


if __name__ == "__main__":
    test_concat_dataset()
