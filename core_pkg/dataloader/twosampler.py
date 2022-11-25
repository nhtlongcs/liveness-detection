# Source:
# https://github.com/HiLab-git/SSL4MIS/blob/8ebc2d9e3455d01ee2b47a9379aeaf213d72570c/code/dataloaders/dataset.py
# https://github.com/kaylode/ivos/blob/master/source/cps/datasets/twostreamloader.py
import itertools
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler
import torch.utils.data as data
from torch.utils.data import ConcatDataset as ConcatDataset


class TwoStreamBatchSampler(BatchSampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, primary_batch_size,
                 secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = primary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (primary_batch + secondary_batch
                for (primary_batch, secondary_batch) in zip(
                    grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):

    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamDataLoader(torch.utils.data.DataLoader):
    """
    Two streams for dataset. Separate batch with additional `split point` 
    """

    def __init__(self, dataset_l: torch.utils.data.Dataset,
                 dataset_u: torch.utils.data.Dataset, batch_sizes: List[int],
                 **kwargs) -> None:
        self.dataset_l = dataset_l
        self.dataset_u = dataset_u
        # self.classnames = dataset_l.classnames
        cat_dataset = ConcatDataset([dataset_l, dataset_u])
        total_length = len(dataset_l) + len(dataset_u)
        labeled_idxs = list(range(0, len(dataset_l)))
        unlabeled_idxs = list(range(len(dataset_l), total_length))
        self.batch_sizes = batch_sizes

        # self.num_classes = dataset_l.num_classes
        # self._encode_masks = dataset_l._encode_masks
        sampler = TwoStreamBatchSampler(primary_indices=labeled_idxs,
                                        secondary_indices=unlabeled_idxs,
                                        primary_batch_size=batch_sizes[0],
                                        secondary_batch_size=batch_sizes[1])

        super().__init__(dataset=cat_dataset,
                         collate_fn=self.mutual_collate_fn,
                         batch_sampler=sampler,
                         **kwargs)

    def mutual_collate_fn(self, batch):
        """
        Mutual collate
        """
        batch_dict = {
            "images":
            torch.stack([x['image'] for x in batch]),
            "labels":
            torch.stack([x['label'] for x in batch if 'label' in x], dim=0),
            "filenames": [x['filename'] for x in batch],
            "video_ids": [x['video_id'] for x in batch],
            "frame_ids": [x['frame_id'] for x in batch],
        }
        batch_dict.update({'split_pos': batch_dict['labels'].shape[0]})
        return batch_dict
