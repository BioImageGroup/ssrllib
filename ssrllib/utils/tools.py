import random
from typing import List, Tuple
from PIL import Image

import numpy as np
from openslide import OpenSlide
from pytorch_lightning.utilities.cli import LightningCLI
import torch
from torchvision.transforms import ToPILImage
import numpy.random as rnd
from torchvision.transforms.transforms import ToTensor
from ssrllib.utils import io

to_tensor = ToTensor()


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.Adam)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)


def tile_image(image: Image, size: int) -> List[torch.Tensor]:
    """Extract adjacent tiles of a given size from an image

    Args:
        image (Image): the image to be tiled
        size (int): the size of the tiles

    Returns:
        List[torch.Tensor]: List of resulting tiles
    """

    image_t = to_tensor(image)
    c, w, h = image_t.shape

    image_t = image_t.unfold(1, size, size).unfold(
        2, size, size).reshape(c, -1, size, size)
    image_t = torch.unbind(image_t, dim=1)

    return image_t


def extract_bbox_from_image(slide: OpenSlide, bbox) -> Image:
    """Extract the region identified by a bounding box from a given OpenSlide image

    Args:
        slide (OpenSlide): the OpenSlide image from which the region will be extracted
        bbox ([type]): the bounding box defining the region to extract

    Returns:
        Image: a PIL Image image representing the desired bounding box
    """

    all_x = [pos[0] for pos in bbox]
    all_y = [pos[1] for pos in bbox]

    top_left_x = np.min(all_x)
    top_left_y = np.min(all_y)
    bottom_right_x = np.max(all_x)
    bottom_right_y = np.max(all_y)

    tile = slide.read_region((top_left_x, top_left_y),
                             size=(bottom_right_x - top_left_x,
                                   bottom_right_y - top_left_y),
                             level=0).convert('RGB')

    return tile


def check_bbox_size(bbox: Tuple, size: int) -> bool:
    """Check whether the bounding box is at least as big as our desired size (in both dimensions)

    Args:
        bbox (Tuple): the bounding box to be checked
        size (int): the minimum size

    Returns:
        bool: True if the bounding box contains a region of at least (size, size) shape, False otherwise
    """

    if bbox[1][0] - bbox[0][0] > size and bbox[2][1] - bbox[1][1] > size:
        return True
    return False


def jigsaw_tile(batch: torch.Tensor):
    assert batch.ndim == 3, f'batch should have 3 dimensions, got shape {batch.shape}'

    channels, width, height = batch.shape
    assert width == height, NotImplementedError(
        f'Jigsaw is only implemented for square input images for now')

    size = int(width//3)
    tiles = 9

    tiles = batch.unfold(1, size, size).unfold(
        2, size, size).reshape(9, channels, size, size)

    return tiles


def jigsaw_scramble(tiled_batch, permutations):
    """
    Scrambles the batch tiles with some pseudo-random permutations
    Args:
        tiles (torch.tensor): Batch of tiles of shape (T, B, C, W, H)

    Returns:
        (torch.tensor): Batch of scrambled tiles of shape (T, B, C, W, H)
    """
    # Extract random permutation from list and store it
    perm_idx = rnd.choice(permutations.shape[0])
    perm = permutations[perm_idx]

    # tile indices in the permutations (t's) are 1-indexed
    scrambled_tiles = torch.stack([tiled_batch[t-1] for t in perm])

    # the permutation index acts as the class to be predicted
    return scrambled_tiles, perm_idx


def get_bbox_from_path(roi: Tuple) -> List[Tuple[int, int]]:
    """Extract the bounding box coordinates from a path Element

    Args:
        roi (Tuple): the path Element to be converted to bounding box format

    Returns:
        List[Tuple[int, int]]: list of tuples identifying the vertiecs of the bounding box (with no specific order)
    """

    bbox = []
    for coord in roi[0]:
        bbox.append((int(coord.get('X').split('.')[0]), int(
            coord.get('Y').split('.')[0])))

    return bbox


def tile_rois(images, rois, labels, size: int):
    """
    Given the images and the xml file defining all the ROIs, return a list of tiles of a given size, extracted from
    the images.

    :return: A list of tiles of a given dimension.
    """
    tiled_images = []
    tiled_labels = []
    slides_reference = []
    rois_reference = []

    for image_idx, (image, rois, label) in enumerate(zip(images, rois, labels)):

        for roi_idx, roi in enumerate(rois):

            # Extract bounding box from xml path coordinates
            bbox = get_bbox_from_path(roi)

            # Skip if bbox is not big enough
            if not check_bbox_size(bbox, size):
                continue

            crop = extract_bbox_from_image(image, bbox)
            tiles = tile_image(crop, size)
            labels = [label, ] * len(tiles)
            slides_ref = [image_idx, ] * len(tiles)
            rois_ref = [roi_idx, ] * len(tiles)

            tiled_images.extend(tiles)
            tiled_labels.extend(labels)
            slides_reference.extend(slides_ref)
            rois_reference.extend(rois_ref)

            # create entry for image #image_idx that says that images from "seen_rois" to "seen_rois + len(tiles)"
            # correspond to patient #image_idx

    return torch.stack(tiled_images), np.array(tiled_labels), (slides_reference, rois_reference)
