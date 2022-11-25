from core_pkg.dataloader import TwoStreamDataLoader
from core.dataset import DATASET_REGISTRY
from core.augmentations import TRANSFORM_REGISTRY
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import pytest


@pytest.mark.order(1)
def test_dual_dataset(dataset_name="ImageFolderFromCSV"):
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

    dataloader = TwoStreamDataLoader(
        dataset_l=ds,
        dataset_u=uds,
        batch_sizes=[2, 2],
        shuffle=False,
        num_workers=0,
    )
    for i, batch in enumerate(dataloader):
        print(batch.keys())
        print(batch["images"].shape)
        print(batch["labels"].shape)
        if i >= 5:
            break


if __name__ == "__main__":
    test_dual_dataset()
