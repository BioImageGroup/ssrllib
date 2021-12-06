import pytorch_lightning as pl
from torch.utils.data import DataLoader
from ssrllib.data.datasets import *
from numpy.linalg import norm
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY


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


@DATAMODULE_REGISTRY
class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_train: MultiFileClassificationDataset,
                 dataset_val: MultiFileClassificationDataset,
                 dataset_test: MultiFileClassificationDataset,
                 batch_size: int,
                 workers: int):

        super().__init__()

        self.batch_size = batch_size
        self.workers = workers
        self.ds_train = dataset_train
        self.ds_val = dataset_val
        self.ds_test = dataset_test

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.workers)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=1, num_workers=self.workers)


@DATAMODULE_REGISTRY
class PredictionDataModule(pl.LightningDataModule):
    def __init__(self, dataset_pred: MultiFileClassificationDataset, batch_size: int, workers: int) -> None:
        super().__init__()

        self.ds_pred = dataset_pred
        self.batch_size = batch_size
        self.workers = workers

    def predict_dataloader(self):
        return DataLoader(self.ds_pred, batch_size=self.batch_size, num_workers=self.workers)
