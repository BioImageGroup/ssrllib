from typing import Tuple

import numpy as np
import torch
from scipy.ndimage import rotate
import numpy.random as rnd

class PretextRotation(object):
    def __init__(self, base_angle: int, multiples: int):
        """
        Apply a random rotation of (k * base) degrees, where k = {0, 1, 2, 3}. It also returns the degrees along with the
        rotated image.

        N.B. Since this adds a return value along with the image, it cannot be composed with other transforms.

        :param base_angle: base angle for the rotation
        :param multiples: the possible value of k are in range(multiples)
        """
        self.base_angle = base_angle
        self.multiples = multiples

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        mult = np.random.randint(self.multiples)

        if mult > 0:
            x = torch.Tensor(rotate(x, angle=self.base_angle * mult, axes=(1, 2)))

        return x, mult


class MinMaxNormalize():
    def __init__(self, min=0, max=1):
        self.min = min
        self.max = max

    def __call__(self, x):
        m = x.min()
        M = x.max()
        eps = 1e-5
        return (x - m + eps) / (M - m + eps)


class Flip(object):
    """
    Perform a flip along a specific dimension

    :param initialization prob: probability of flipping
    :param initialization dim: (spatial) dimension to flip (excluding the channel dimension, default flips last dim)
    :param forward x: input array (C, N_1, N_2, ...)
    :return: output array (C, N_1, N_2, ...)
    """

    def __init__(self, prob=0.5, dim=None):
        self.prob = prob
        if dim is None:
            self.dim = -1
        else:
            self.dim = dim + 1

    def __call__(self, x):
        if rnd.rand() < self.prob:
            return torch.flip(x, dims=(self.dim,))
        else:
            return x


class Rotate90(object):
    """
    Rotate the inputs by 90 degree angles

    :param initialization prob: probability of rotating
    :param forward x: input array (C, N_1, N_2, ...)
    :return: output array (C, N_1, N_2, ...)
    """

    def __init__(self, prob=1):
        self.prob = prob

    def __call__(self, x):

        if rnd.rand() < self.prob:
            n = x.ndim
            return torch.rot90(x, k=rnd.randint(0, 4), dims=[n - 2, n - 1])
        else:
            return x


class ContrastAdjust(object):
    """
    Apply contrast adjustments to the data

    :param initialization prob: probability of adjusting contrast
    :param initialization adj: maximum adjustment (maximum intensity shift for minimum and maximum the new histogram)
    :param forward x: input array (N_1, N_2, N_3, ...)
    :return: output array (N_1, N_2, N_3, ...)
    """

    def __init__(self, prob=1, adj=0.1):
        self.prob = prob
        self.adj = adj

    def __call__(self, x):

        if rnd.rand() < self.prob:
            x_ = x

            m = torch.min(x_)
            M = torch.max(x_)
            r1 = rnd.rand()
            r2 = rnd.rand()
            m_ = 2 * self.adj * r1 - self.adj + m
            M_ = 2 * self.adj * r2 - self.adj + M

            if m != M:
                return ((x - m) / (M - m)) * (M_ - m_) + m_
            else:
                return x
        else:
            return x


class AddNoise(object):
    """
    Adds noise to the input

    :param initialization prob: probability of adding noise
    :param initialization sigma_min: minimum noise standard deviation
    :param initialization sigma_max: maximum noise standard deviation
    :param forward x: input array (N_1, N_2, ...)
    :return: output array (N_1, N_2, ...)
    """

    def __init__(self, prob=0.5, sigma_min=0.0, sigma_max=1.0):
        self.prob = prob
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, x):

        if rnd.rand() < self.prob:
            sigma = rnd.uniform(self.sigma_min, self.sigma_max)

            noise = rnd.randn(*x.shape) * sigma
            return (x + noise).float()
        else:
            return x
