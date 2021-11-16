import torch.nn as nn


########################################
# ---------- Wrapped Losses ---------- #
########################################

class CELoss:
    def __init__(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def __call__(self, preds, labels, w=None):
        return self.loss_fn(preds.double(), labels.long())


class MSELoss:
    def __init__(self):
        self.loss_fn = nn.MSELoss()

    def __call__(self, preds, labels):
        return self.loss_fn(preds, labels)
