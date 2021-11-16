import random
from typing import List

import numpy as np
import torch
from torchvision.transforms import ToPILImage
import numpy.random as rnd

def batch_to_image(batch: List) -> List:
    images = []
    for tensor in batch:
        images.append(float_tensor_to_uint_array(tensor))

    return images


def float_tensor_to_uint_array(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = np.moveaxis(array, 0, -1) * 255
    array = array.astype(np.uint8)

    return array


def set_seed(seed: int):
    """
    Seeds all stochastic processes for reproducibility

    :param seed: Seed to apply to all processes
    """

    # NumPy
    np.random.seed(seed)
    # Random
    random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def jigsaw_tile(batch: torch.Tensor):
    assert batch.ndim == 3, f'batch should have 3 dimensions, got shape {batch.shape}'
    
    channels, width, height = batch.shape
    assert width == height, NotImplementedError(f'Jigsaw is only implemented for square input images for now')
    
    size = int(width//3)
    tiles = 9
    
    tiles = batch.unfold(1, size, size).unfold(2, size, size).reshape(9, channels, size, size)
    # tiles = tiles.view(1, 9, 3, 42, 42)
    # tiles = tiles.view(9, batch_size, , )
    # .view(9, batch_size, channels, size, size)
    # print(tiles.shape)
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

