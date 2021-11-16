import builtins
import os
import xml.etree.ElementTree as et
from abc import abstractmethod
from glob import glob
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from openslide import OpenSlide
from ssrllib.util.augmentation import PretextRotation
from ssrllib.util.tools import jigsaw_scramble, jigsaw_tile
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, RandomCrop, Compose

to_tensor = ToTensor()


def _load_images(images: List[str], type: str) -> List:
    """
    Open a list of image files from filesystem

    :param images: list of images files
    :param type: extension of the images

    :return: list of open whole-slide images
    """
    slides = []

    for image in images:
        if type == 'ndpi':
            slides.append(OpenSlide(image))
        elif type == 'svs':
            slide = OpenSlide(image)

            slides.append(OpenSlide(image))
        else:
            NotImplementedError(f'The required format {type} is not implemented yet')

    return slides


def _load_rois(xmls: List[str]) -> List:
    """
    Open a list of xml files and extract the coordinates for the Regions Of Interest

    :param xmls: list of xml files
    :return: list of xml Elements
    """
    rois = []
    for xml in xmls:
        tree = et.parse(xml)
        root = tree.getroot()
        rois.append(root[0])

    return rois


def _get_bbox_from_path(roi) -> List[Tuple[int, int]]:
    """
    Extract the bounding box coordinates from a path Element

    :param roi: the path Element to be converted to bounding box format
    :return: list of tuples identifying the vertiecs of the bounding box (with no specific order)
    """

    bbox = []
    for coord in roi[0]:
        bbox.append((int(coord.get('X').split('.')[0]), int(coord.get('Y').split('.')[0])))

    return bbox


def _check_bbox_size(bbox, size: int) -> bool:
    """
    Check whether the bounding box is at least as big as our desired size (in both dimensions)

    :param bbox: the bounding box to be checked
    :param size: the minimum size
    :return: True if the bounding box contains a region of at least (size, size) shape, False otherwise
    """
    if bbox[1][0] - bbox[0][0] > size and bbox[2][1] - bbox[1][1] > size:
        return True
    return False


def _extract_bbox_from_image(slide: OpenSlide, bbox) -> Image:
    """
    Extract the region identified by a bounding box from a given OpenSlide image

    :param slide: the OpenSlide image from which the region will be extracted
    :param bbox: the bounding box defining the region to extract
    :return: a PIL Image image representing the desired bounding box
    """
    all_x = [pos[0] for pos in bbox]
    all_y = [pos[1] for pos in bbox]

    top_left_x = np.min(all_x)
    top_left_y = np.min(all_y)
    bottom_right_x = np.max(all_x)
    bottom_right_y = np.max(all_y)

    tile = slide.read_region((top_left_x, top_left_y),
                             size=(bottom_right_x - top_left_x, bottom_right_y - top_left_y),
                             level=0).convert('RGB')

    return tile


def _tile_image(image: Image, size: int) -> List[torch.Tensor]:
    """
    Extract adjacent tiles of a given size from an image

    :param image: the image to be tiled
    :param size: the size of the tiles

    :return: List of resulting tiles
    """

    image_t = to_tensor(image)
    c, w, h = image_t.shape

    image_t = image_t.unfold(1, size, size).unfold(2, size, size).reshape(c, -1, size, size)
    image_t = torch.unbind(image_t, dim=1)

    return image_t


class MultiFileClassificationDataset(Dataset):
    """
    Base class for a dataset made of multiple classes and files. The structure this Class is intendend to cover is as
    follows:
    .
    |-- "data_dir"
    |     |-- "00_<ClassName-0>"
    |     |           |-- "<Image-ID#_...>"
    |     |           |-- "<Image-ID#_...>"
    |     |
    |     |-- "01_<ClassName-1>"
    |     |           |-- "<Image-ID#_...>"
    |     |           |-- "<Image-ID#_...>"

    """

    def _get_class_files(self, type: str = 'ndpi') -> List[str]:
        """
        Gathers all files of a given type from the filesystem, for each available class. Files are sorted alphanumerially
        to increase robustness.

        :param type: Extention of files to be collected
        :return: List of paths to the images
        """
        files = []
        for class_dir in sorted(glob(os.path.join(self.data_dir, f"*"))):
            files.extend(sorted(glob(os.path.join(class_dir, f"*.{type}"))))

        return files

    def _get_class_labels(self, type) -> List[int]:
        """
        Gathers all class labels from folder structure

        :return: List of class indices
        """

        classes = []

        for class_dir in sorted(glob(os.path.join(self.data_dir, f"*"))):
            class_idx = int(class_dir.split('/')[-1].split('_')[0])
            classes.extend([class_idx] * len(glob(os.path.join(class_dir, f"*.{type}"))))

        return np.array(classes)

    def __init__(self, data_dir: str, type: str, split: Tuple[float, float], stratified: bool = False):
        """

        :param data_dir: Root folder for the multi-file dataset
        :param type: Extension of the files
        """

        self.data_dir = data_dir
        self.type = type

        self.image_files = self._get_class_files(type=self.type)
        self.labels = self._get_class_labels(type=self.type)
        self.images = _load_images(self.image_files, type=self.type)
        self.split = split
        self.stratified = stratified

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
            tiles = _tile_image(image, size)
            labels = [label, ] * len(tiles)

            tiled_images.extend(tiles)
            tiled_labels.extend(labels)
        
        return torch.stack(tiled_images), np.array(tiled_labels)

    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[float, float], stratified: bool = False, transforms: Compose = Compose([])):
        super(MultiFileStandardClassificationDataset, self).__init__(data_dir, type, split, stratified)

        self.images, self.labels = self._tile_images(size)

        self.split_data()
        self.crop = Compose([RandomCrop(size=size)])
        self.transforms = transforms

    def __getitem__(self, item):
        return self.transforms(self.crop(self.images[item])), self.labels[item]




class MultiFileROIClassificationDataset(MultiFileClassificationDataset):
    """
    This class is an extension of MultiFilClassificationDataset, covering the possibility of having XML files for Region
    Of Interest definition.
    """

    def _tile_rois(self, size: int):
        """
        Given the images and the xml file defining all the ROIs, return a list of tiles of a given size, extracted from
        the images.

        :return: A list of tiles of a given dimension.
        """
        tiled_images = []
        tiled_labels = []

        for image, rois, label in zip(self.images, self.rois, self.labels):

            for roi in rois:

                # Extract bounding box from xml path coordinates
                bbox = _get_bbox_from_path(roi)

                # Skip if bbox is not big enough
                if not _check_bbox_size(bbox, size):
                    continue

                crop = _extract_bbox_from_image(image, bbox)
                tiles = _tile_image(crop, size)
                labels = [label, ] * len(tiles)

                tiled_images.extend(tiles)
                tiled_labels.extend(labels)

        return torch.stack(tiled_images), np.array(tiled_labels)

    def split_data(self):
        if self.stratified:
            self.split_data_stratified()
        else:
            self.split_data_sorted()

    def split_data_sorted(self):
        start = int(len(self) * self.split[0])
        end = int(len(self) * self.split[1])
        self.images = self.images[start:end]
        self.labels = self.labels[start:end]

    def split_data_stratified(self):
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

    def class_distr(self):
        print(np.bincount(self.labels))

    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[int, int], stratified: bool = False, transforms: Compose = Compose([])):
        """

        :param data_dir: Root folder for the multi-file dataset
        :param type: Extension of the files
        :param size: Size of the tiles to be extracted
        """

        super().__init__(data_dir, type, split, stratified)
        self.xmls = self._get_class_files(type='xml')
        self.rois = _load_rois(self.xmls)

        self.images, self.labels = self._tile_rois(size)
        print(f'class distr {self.class_distr()}')

        self.split_data()
        self.crop = Compose([RandomCrop(size=size)])
        self.transforms = transforms

    def __getitem__(self, item):
        return self.transforms(self.crop(self.images[item])), self.labels[item]


class ClassificationROIDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[int, int], transforms: Compose = Compose([]), stratified: bool = False):
        super(ClassificationROIDataset, self).__init__(data_dir, type, size, split, stratified, transforms)


class ClassificationDataset(MultiFileStandardClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[int, int], transforms: Compose = Compose([]), stratified: bool = False):
        super(ClassificationDataset, self).__init__(data_dir, type, size, split, stratified, transforms)


class RotationDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[int, int], transforms: Compose = Compose([]),
                 base_angle: int = 90, multiples: int = 4):
        super(RotationDataset, self).__init__(data_dir, type, size, split, transforms)

        self.rotation = PretextRotation(base_angle, multiples)

    def __getitem__(self, item):
        image = self.crop(self.transforms(self.images[item]))
        image, label = self.rotation(image)

        return image, label


class AutoencodingROIDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[int, int], transforms: Compose = Compose([]), stratified: bool = False):
        super(AutoencodingROIDataset, self).__init__(data_dir, type, size, split, stratified, transforms)

    def __getitem__(self, item):
        image = self.crop(self.transforms(self.images[item]))

        return image, image


class JigsawROIDataset(MultiFileROIClassificationDataset):
    def __init__(self, data_dir: str, type: str, size: int, split: Tuple[int, int], permutations: str, transforms: Compose = Compose([]), stratified: bool = False):
        super(JigsawROIDataset, self).__init__(data_dir, type, size, split, stratified, transforms)

        self.permutations = np.load(permutations)

    def __getitem__(self, item):
        image = self.crop(self.transforms(self.images[item]))
        image = jigsaw_tile(image)
        image, label = jigsaw_scramble(image, self.permutations)
        return image, label
