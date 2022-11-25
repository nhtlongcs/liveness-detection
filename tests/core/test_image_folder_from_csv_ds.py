from core.dataset import DATASET_REGISTRY
from core.augmentations import TRANSFORM_REGISTRY
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import pytest


@pytest.mark.order(1)
def test_vision_dataset(dataset_name="ImageFolderFromCSV"):
    image_transform = TRANSFORM_REGISTRY.get('train_classify_tf')(img_size=380)

    ds = DATASET_REGISTRY.get(dataset_name)(
        CSV_PATH="data/train/labels_keyframes_test.csv",
        IMG_DIR="data/train/keyframes",
        transform=image_transform)
    dataloader = DataLoader(
        ds,
        collate_fn=ds.collate_fn,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    for i, batch in enumerate(dataloader):
        print(batch["images"].shape)
        print(batch["labels"].shape)
        if i >= 5:
            break
