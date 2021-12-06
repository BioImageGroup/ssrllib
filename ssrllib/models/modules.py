from imblearn.base import SamplerMixin
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV
from ssrllib.utils.tools import jigsaw_tile, jigsaw_scramble
from ssrllib.utils.io import print_ts
from typing import Any, Dict, List, Sequence, Tuple, Union
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torchmetrics import Metric
from pytorch_lightning.utilities.cli import MODEL_REGISTRY


class BaseModule(LightningModule):
    def __init__(self):
        super(BaseModule, self).__init__()
        pass

    # def configure_optimizers(self):
    #     pass

    def load_from_pretext(self, ckpt_path, drop_head, freeze_backbone):
        # Laod state dict from filesystem
        try:
            state_dict = torch.load(ckpt_path)['state_dict']
        except KeyError:
            state_dict = torch.load(ckpt_path)

        # Drop head parameters from state dict
        if drop_head:
            state_dict = {name: param for name,
                          param in state_dict.items() if not name.startswith('head')}

        # Now load the remaining parameters
        self.load_state_dict(state_dict, strict=False)

        # freeze the backbone network layers
        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False

    def track_val_images(self, imgs: torch.Tensor, preds: Sequence):
        """Keep track of the last image and prediction of the validation loop
        This is done to be compatible with the custom ValidationVisualization
        callback, that plots images and reconstructions (in case the task is
        autoencoding) at the end of the validation step

        Args:
            imgs (torch.Tensor): labels
            preds (Sequence): validation images
            labels (Sequence): validation predictions
        """

        retained_imgs = []
        retained_preds = []
        for idx in self.random_idx_for_plotting:
            retained_imgs.append(imgs[idx])
            retained_preds.append(preds[idx])

        self.retained = retained_imgs + retained_preds


@MODEL_REGISTRY
class TrainingModule(BaseModule):
    def __init__(self, backbone: nn.Module, head: nn.Module, loss: nn.Module, metric: Metric, input_shape: Sequence[int], jigsaw: bool = False):
        super().__init__()

        self.example_input_array = torch.zeros((1,) + tuple(input_shape))

        self.backbone = backbone
        self.head = head
        self.loss_module = loss
        self.metric = metric

        self.jigsaw = jigsaw

        # TODO: Move to another location that does not dirty loo[] code
        self.random_idx_for_plotting = np.random.randint(0, 5, size=5)
        # self.save_hyperparameters()

    def forward(self, x):
        if self.jigsaw:
            return self.siamese_forward(x)
        else:
            return self.standard_forward(x)

    def standard_forward(self, x):
        x = self.backbone(x)
        x = self.head(x)

        return x

    def siamese_forward(self, x):
        outputs = []
        for i in range(x.shape[1]):
            outputs.append(self.backbone(x[:, i]))

        x = torch.cat(outputs, 1)
        x = self.head(x)

        return x

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)

        self.log(f"train/metric", metric, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"train/loss", loss, on_step=True, on_epoch=True)

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)

        self.track_val_images(imgs, preds)

        self.log(f"val/metric", metric, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"val/loss", loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)

        self.log(f"test/metric", metric, on_step=True, on_epoch=True)
        self.log(f"test/loss", loss, on_step=True, on_epoch=True)


@MODEL_REGISTRY
class PredictionModule(BaseModule):
    def __init__(self,
                 backbone: nn.Module,
                 head: ClassifierMixin,
                 splits: Dict[str, List[float]],
                 input_shape: Tuple[int, int, int],
                 samplers: List[Tuple[str, SamplerMixin]],
                 param_grid: Dict[str, List[Union[int, float, str]]],
                 grid_search_params: Dict[str, Union[int, float, str]],
                 ckpt_path: str = None,
                 metta: bool = False):
        super().__init__()

        self.example_input_array = torch.zeros((1,) + tuple(input_shape))

        self.backbone = backbone
        self.head = head

        self.splits = splits
        self.samplers = samplers
        self.param_grid = param_grid
        self.grid_search_params = grid_search_params
        self.ckpt_path = ckpt_path
        self.metta = metta

        self.pipeline = self._setup_pipeline()
        self.gscv = self._setup_gridsearch()

        if self.ckpt_path:
            self.load_from_pretext(self.ckpt_path, drop_head=True, freeze_backbone=True)

    def _setup_pipeline(self):
        # pipeline = Pipeline(self.samplers + [('classifier', self.head)])
        pipeline = Pipeline([('classifier', self.head)])
        return pipeline

    def _setup_gridsearch(self):
        gscv = GridSearchCV(self.pipeline, param_grid=self.param_grid,
                            **self.grid_search_params)
        return gscv

    def forward(self, x):
        return self.backbone(x)

    def predict_step(self, batch, batch_idx):
        if self.metta:
            return self.MeTTA_predict_step(batch, batch_idx)
        else:
            return self.standard_predict_step(batch, batch_idx)

    def standard_predict_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)

        return preds, labels

    # TODO: Implement averagin of embeddings
    def MeTTA_predict_step(self, batch, batch_idx):
        imgs, labels = batch

        embeddings = []
        for aug in imgs:
            embeddings.append(self(aug))

        mean_embeddings = torch.cat(embeddings, dim=0).mean(dim=0).reshape(1, -1)

        return mean_embeddings, labels
