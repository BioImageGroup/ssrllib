import sys

from ssrllib.models.blocks import *
from ssrllib.models.heads import *
from ssrllib.models.models import *
from ssrllib.models.resnet import ResNet
from ssrllib.util.metric import *
from ssrllib.util.loss import *
from ssrllib.util.augmentation import RandomFlip, RandomRotate90, ContrastAdjust, AddNoise
from ssrllib.data.base import *
from ssrllib.data.base import MeTTADataset

from torch.nn import ReLU, LeakyReLU, GELU, Tanh
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import *
from pytorch_lightning.callbacks import *
from albumentations.augmentations.crops import RandomResizedCrop


def create_module(hparams):
    return getattr(sys.modules[__name__], hparams.pop(f'name'))(**hparams)
