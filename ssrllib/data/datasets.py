from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms.transforms import ToPILImage, ToTensor
from ssrllib.utils.augmentations import CustomAugmentation, PretextRotation
from ssrllib.utils.tools import jigsaw_scramble, jigsaw_tile
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, Compose, Resize, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip
from ssrllib.utils import io
from ssrllib.utils import tools
from ssrllib.utils.io import print_ts

import albumentations as A


class MultiFileClassificationDataset(Dataset):

    def _split_data(self):
        print_ts(
            f'starting class distribution: {np.bincount(self.labels)} ({np.unique(self.labels)} classes)')
        print_ts(f'Splitting -> {(self.splitting).upper()} (portion: {self.split})')
        if self.splitting == 'stratified':
            self._split_data_stratified()
        elif self.splitting == 'uniform':
            self._split_data_uniform()
        elif self.splitting == 'sorted':
            self._split_data_sorted()
        else:
            NotImplementedError(
                f'Available splitting methods are stratified/uniform/sorted, got {self.splitting}')

        print_ts(
            f'final class distribution: {np.bincount(self.labels)} ({np.unique(self.labels)} classes)')

    def _split_data_sorted(self):
        start = int(len(self) * self.split[0])
        end = int(len(self) * self.split[1])
        self.images = self.images[start:end]
        self.labels = self.labels[start:end]

    def _split_data_uniform(self):
        class_dist = np.bincount(self.labels)
        smallest_class_size = np.min(class_dist)

        labels_split = []
        images_split = []

        start = int(smallest_class_size * self.split[0])
        end = int(smallest_class_size * self.split[1])

        for cls, _ in enumerate(class_dist):
            idx_split = self.labels == cls
            labels_split.extend(self.labels[idx_split][start:end])
            images_split.extend(self.images[idx_split][start:end])

        self.labels = labels_split
        self.images = images_split

    def _split_data_stratified(self):
        class_dist = np.bincount(self.labels)

        labels_split = []
        images_split = []

        for cls, size in enumerate(class_dist):
            start = int(size * self.split[0])
            end = int(size * self.split[1])

            idx_split = self.labels == cls
            labels_split.extend(self.labels[idx_split][start:end])
            images_split.extend(self.images[idx_split][start:end])

        self.labels = labels_split
        self.images = images_split

    def __init__(self, data_dir: str, type: str, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = []):
        """Base class for a dataset made of multiple classes and files.

        The structure this Class is intendend to cover is as follows:
            .
            |-- "data_dir"
            |     |-- "00_<ClassName-0>"
            |     |           |-- "<Image-ID#_...>"
            |     |           |-- "<Image-ID#_...>"
            |     |
            |     |-- "01_<ClassName-1>"
            |     |           |-- "<Image-ID#_...>"
            |     |           |-- "<Image-ID#_...>"

         Args:
            data_dir (str): The root directory of the data
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """

        self.data_dir = data_dir
        self.type = type

        self.image_files = io.get_class_files(self.data_dir, type=self.type)
        self.labels = io.get_class_labels(self.data_dir, type=self.type)
        self.images = io.load_images(self.image_files, type=self.type)
        self.split = split
        self.splitting = splitting
        self.transforms = Compose(transforms)

    @abstractmethod
    def __getitem__(self, item):
        pass

    def __len__(self):
        return len(self.images)


class MultiFileStandardClassificationDataset(MultiFileClassificationDataset):

    def _tile_images(self, size: int):
        tiled_images = []
        tiled_labels = []

        for image, label in zip(self.images, self.labels):
            tiles = tools.tile_image(np.array(image), size)
            labels = [label, ] * len(tiles)

            tiled_images.extend(tiles)
            tiled_labels.extend(labels)

        return torch.stack(tiled_images), np.array(tiled_labels)

    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = []):
        """This class is an extension of MultiFilClassificationDataset, covering the possibility of having XML files for ROIs definition.

         Args:
            data_dir (str): The root directory of the data
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """
        super(MultiFileStandardClassificationDataset, self).__init__(
            data_dir, type, split, splitting, transforms)

        self.images, self.labels = self._tile_images(size)
        self._split_data()

    def __getitem__(self, item):
        return self.transforms(self.images[item]), self.labels[item]


class MultiFileROIClassificationDataset(MultiFileClassificationDataset):

    def class_distr(self):
        return np.bincount(self.labels)

    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = []):
        """This class is an extension of MultiFilClassificationDataset, covering the possibility of having XML files for ROIs definition.

         Args:
            data_dir (str): The root directory of the data
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """
        super().__init__(data_dir, type, split, splitting, transforms)
        self.xmls = io.get_class_files(self.data_dir, type='xml')
        self.rois = io.load_rois(self.xmls)

        self.images, self.labels, self.patient_reference = tools.tile_rois(
            self.images, self.rois, self.labels, size)

        self._split_data()
        # self.crop = Compose([RandomCrop(size=size), Normalize(0, 1)])
        self.crop = Compose([RandomCrop(size=size)])

    def __getitem__(self, item):
        return self.transforms(self.images[item]), self.labels[item]

    def get_patient_reference(self):
        return self.patient_reference


class ClassificationROIDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = []):
        """Just a name wrapper for a long class name

        Args:
            data_dir (str): The root directory of the data
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """
        super().__init__(data_dir, type, size, split, splitting, transforms)


class ClassificationDataset(MultiFileStandardClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = []):
        """Just a name wrapper for a long class name

        Args:
            data_dir (str): The root directory of the data
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """
        super(ClassificationDataset, self).__init__(
            data_dir, type, size, split, splitting, transforms)


class RotationROIDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = [],
                 base_angle: int = 90, multiples: int = 4):
        """Dataset for autonecoding training. Loads Images and performs the intended rotation before returning them

        Args:
            data_dir (str): The root directory of the data
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
            base_angle (int, optional): Base angle of rotation. Defaults to 90.
            multiples (int, optional): Number of possible multiples of the angle uxed for rotation. Defaults to 4.
        """
        super(RotationROIDataset, self).__init__(
            data_dir, type, size, split, splitting, transforms)

        self.rotation = PretextRotation(base_angle, multiples)

    def __getitem__(self, item):
        image = self.transforms(self.images[item])
        image, label = self.rotation(image)

        return image, label


class RotationDataset(MultiFileStandardClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = [],
                 base_angle: int = 90, multiples: int = 4):
        """Dataset for autonecoding training. Loads Images and performs the intended rotation before returning them

        Args:
            data_dir (str): The root directory of the data
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
            base_angle (int, optional): Base angle of rotation. Defaults to 90.
            multiples (int, optional): Number of possible multiples of the angle uxed for rotation. Defaults to 4.
        """
        super(RotationDataset, self).__init__(
            data_dir, type, size, split, splitting, transforms)

        self.rotation = PretextRotation(base_angle, multiples)

    def __getitem__(self, item):
        image = self.transforms(self.images[item])
        image, label = self.rotation(image)

        return image, label


class AutoencodingROIDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = []):
        """Dataset for autonecoding training. Loads Images and ROIs and returns them

        Args:
            data_dir (str): The root directory of the data 
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """

        super(AutoencodingROIDataset, self).__init__(
            data_dir, type, size, split, splitting, transforms)

    def __getitem__(self, item):
        image = self.transforms(self.images[item])

        return image, image


class AutoencodingDataset(MultiFileStandardClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted', transforms: List[CustomAugmentation] = []):
        """Dataset for autonecoding training. Loads Images and returns them



        Args:
            data_dir (str): The root directory of the data 
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.

        """
        super(AutoencodingDataset, self).__init__(
            data_dir, type, size, split, splitting, transforms)

    def __getitem__(self, item):
        image = self.transforms(self.images[item])

        return image, image


class JigsawROIDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], permutations: str, transforms: List[CustomAugmentation] = [], splitting: str = 'sorted'):
        """Dataset for jigsaw training. Loads Images and ROIs and performs the intended tiling before returning them

        Args:
            data_dir (str): The root directory of the data 
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            permutations (str): path of file containing pseudo-random permutations.
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """
        super(JigsawROIDataset, self).__init__(
            data_dir, type, size, split, splitting, transforms)

        self.permutations = np.load(permutations)

    def __getitem__(self, item):
        image = self.transforms(self.images[item])
        image = jigsaw_tile(image)
        image, label = jigsaw_scramble(image, self.permutations)
        return image, label


class JigsawDataset(MultiFileStandardClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], permutations: str, transforms: List[CustomAugmentation] = [], splitting: str = 'sorted'):
        """Dataset for jigsaw training. Loads Images and performs the intended tiling before returning them

        Args:
            data_dir (str): The root directory of the data 
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            permutations (str): path of file containing pseudo-random permutations.
            transforms (Compose, optional): transforms.Compose object containing data augmentation. Defaults to Compose([]).
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """
        super(JigsawDataset, self).__init__(
            data_dir, type, size, split, splitting, transforms)

        self.permutations = np.load(permutations)

    def __getitem__(self, item):
        image = self.transforms(self.images[item])
        image = jigsaw_tile(image)
        image, label = jigsaw_scramble(image, self.permutations)
        return image, label


class MeTTAStandardDataset(MultiFileClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted'):
        """This Dataset implements the Meat Test Time approach to generate mean embeddings at test time 

        Args:
            data_dir (str): The root directory of the data 
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """
        super().__init__(data_dir, type, size, split, splitting)

        self.transforms = [
            Compose([ToTensor(), RandomResizedCrop(size), RandomHorizontalFlip(
                p=0.5), RandomVerticalFlip(p=0.5)]),
            Compose([ToTensor(), RandomResizedCrop(size), RandomHorizontalFlip(
                p=0.5), RandomVerticalFlip(p=0.5)]),
            Compose([ToTensor(), RandomResizedCrop(size), RandomHorizontalFlip(
                p=0.5), RandomVerticalFlip(p=0.5)]),
        ]

    def __getitem__(self, item):
        augmentations = []
        for transform in self.transforms:
            res = transform(self.images[item])
            augmentations.append(res)

        label = self.labels[item]

        return augmentations, label


class MeTTAROIDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], splitting: str = 'sorted'):
        """This Dataset implements the Meat Test Time approach to generate mean embeddings at test time 

        Args:
            data_dir (str): The root directory of the data 
            type (str): The file extension of the images
            size (int): The input size of the neural network
            split (Tuple[int, int]): Starting and Ending position of the portion of data to read
            splitting (str, optional): Splitting approach to adopt. Defaults to 'sorted'.
        """
        super().__init__(data_dir, type, size, split, splitting)

        self.transforms = [
            Compose([RandomResizedCrop(size), RandomHorizontalFlip(
                p=0.5), RandomVerticalFlip(p=0.5)]),
            Compose([RandomResizedCrop(size), RandomHorizontalFlip(
                p=0.5), RandomVerticalFlip(p=0.5)]),
            Compose([RandomResizedCrop(size), RandomHorizontalFlip(
                p=0.5), RandomVerticalFlip(p=0.5)]),
        ]

    def __getitem__(self, item):
        augmentations = []
        for transform in self.transforms:
            res = transform(self.images[item])
            augmentations.append(res)

        label = self.labels[item]

        return augmentations, label
