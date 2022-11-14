from core.dataset import DATASET_REGISTRY
import torchvision
from torch.utils.data import DataLoader
from pathlib import Path
import pytest


@pytest.mark.order(1)
def test_vision_dataset(dataset_name="ImageFolderFromCSV"):
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((288, 288)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])
    ds = DATASET_REGISTRY.get(dataset_name)(
        CSV_PATH="data_sample/train/labels_keyframes_test.csv",
        IMG_DIR="data_sample/train/keyframes",
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
