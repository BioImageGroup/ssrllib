from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from ssrllib.data.base import *

from ssrllib.util.common import create_module

from numpy.linalg import norm


def to_probs(split):
    """
    Transform the list of split into a set of probabilities
    """
    split /= norm(split, ord=1)

    return split


def make_splits(total, split):
    train = int(total * split['train'])
    val_test = total - train
    val = int(total * split['val'])
    test = total - train - val

    return [train, val_test], [val, test]


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_hparams: dict, batch_sizes: dict, workers: dict) -> None:
        """
        PytorchLightning DataModule for Mesothelioma Annotated Data.

        :param data_dir: Root folder for the multi-file dataset
        :param type: Extension of the files
        :param train_batch_size: batch size for the training dataloader
        :param val_batch_size: batch size for the validation dataloader
        :param test_batch_size: batch size for the testing dataloader
        :param task_name: the task for which we should load the data: rotation/autoencoding/jigsaw/classification
        """
        super().__init__()
        self.dataset_hparams = dataset_hparams
        self.batch_sizes = batch_sizes
        self.workers = workers

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None):
        # Load the dataset

        if stage == 'fit':
            self.ds_train = create_module(self.dataset_hparams['train'])
            self.ds_val = create_module(self.dataset_hparams['val'])
        elif stage is None or stage == 'test':
            self.ds_test = create_module(self.dataset_hparams['test'])

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_sizes['train'], num_workers=self.workers['train'], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_sizes['val'], num_workers=self.workers['val'])

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_sizes['test'], num_workers=self.workers['test'])
