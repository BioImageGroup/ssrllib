import torchmetrics as tm
import torch
from torch import nn
import torch
import torchmetrics as tm
from torch import nn


########################################
# ---------- Wrapped Metrics --------- #
########################################


class Accuracy(object):
    """
    Wrapper around accuracy function

    """

    def __init__(self):
        self.softmax = nn.Softmax(dim=-1)
        self.metric_func = tm.Accuracy().cuda()

    def __call__(self, preds, labels):
        return self.metric_func(torch.argmax(self.softmax(preds), dim=-1), labels)


class MSE(object):
    """
    Wrapper around skimage's mean_squared_error function
    """

    def __init__(self):
        self.metric_func = tm.MeanSquaredError().cuda()

    def __call__(self, *args):
        return self.metric_func(*args)


class mIoU(object):
    """
    Wrapper around mIoU metric function
    """

    def __init__(self):
        self.metric_func = tm.IoU(2).cuda()

    def __call__(self, preds, labels):
        preds = torch.softmax(preds, dim=1)
        return self.metric_func(preds, labels)
