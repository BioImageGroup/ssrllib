import pytorch_lightning as pl
import torch

import ssrllib.util.common
from ssrllib.util.tools import jigsaw_tile, jigsaw_scramble
import numpy as np

class SelfSupervisedModule(pl.LightningModule):
    def __init__(self, backbone_hparams, head_hparams, optimizer_hparams, loss_hparams, scheduler_hparams,
                 metric_hparams, input_shape):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes lr, weight decay, etc.
        """
        super(SelfSupervisedModule, self).__init__()

        self.example_input_array = (1,) + tuple(input_shape)
        if len(tuple(input_shape)) == 4:
            self.jigsaw = True
        else:
            self.jigsaw = False

        # ---------- backbone model ---------- #
        self.backbone = ssrllib.util.common.create_module(backbone_hparams)

        # ---------- head model ---------- #
        # # Create head
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1,) + tuple(input_shape), dtype=torch.float32)

        # out_features = self.backbone(self.example_input_array).numel()
        # head_hparams['in_features'] = out_features
        self.head = ssrllib.util.common.create_module(head_hparams)

        # ---------- optimizer, loss & metric ---------- #
        # Save optimizer params
        self.optimizer_hparams = optimizer_hparams
        self.optimizer_hparams['params'] = self.parameters()

        # Save scheduler params
        self.scheduler_hparams = scheduler_hparams

        # Create loss function
        self.loss_module = ssrllib.util.common.create_module(loss_hparams)

        # Create metric
        self.metric_name = metric_hparams['name']
        self.metric = ssrllib.util.common.create_module(metric_hparams)

        # ---------- everything else ---------- #
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
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

    def configure_optimizers(self):
        optimizer_dict = {}

        # optimizer
        optimizer = ssrllib.util.common.create_module(self.optimizer_hparams)
        self.scheduler_hparams['optimizer'] = optimizer
        optimizer_dict['optimizer'] = optimizer

        # scheduler
        if 'Plateau' in self.scheduler_hparams['name']:
            optimizer_dict['monitor'] = self.scheduler_hparams.pop('monitor')

        scheduler = ssrllib.util.common.create_module(self.scheduler_hparams)
        optimizer_dict['lr_scheduler'] = scheduler

        return optimizer_dict

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log(f"train/{self.metric_name}", metric, prog_bar=True)
        self.log(f"train/loss", loss)

        if batch_idx == 0:
            if labels.ndim > 2:
                self._log_result(preds.detach().cpu().numpy(), prefix='train/preds')
                self._log_result(labels.detach().cpu().numpy(), prefix='train/labels')

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)
        # By default logs it per epoch (weighted average over batches)
        self.log(f"val/{self.metric_name}", metric, prog_bar=True)
        self.log(f"val/loss", loss)

        if batch_idx == 0:
            if labels.ndim > 2:
                self._log_result(preds.detach().cpu().numpy(), prefix='val/preds')
                self._log_result(labels.detach().cpu().numpy(), prefix='val/labels')

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self(imgs)
        loss = self.loss_module(preds, labels)
        metric = self.metric(preds, labels)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log(f"test/{self.metric_name}", metric, prog_bar=True)
        self.log(f"test/loss", loss)

        if batch_idx == 0:
            if labels.ndim > 2:
                self._log_result(preds.detach().cpu().numpy(), prefix='test/preds')
                self._log_result(labels.detach().cpu().numpy(), prefix='test/labels')

    def _log_result(self, preds, prefix='train'):
        # get the tensorboard summary writer
        tensorboard = self.logger.experiment
        tensorboard.add_images(prefix, preds, global_step=self.current_epoch, dataformats='NCHW')

    def load_from_pretext(self, pretrained_model, drop_head, freeze_backbone):
        # Laod state dict from filesystem
        state_dict = torch.load(pretrained_model)['state_dict']

        # Drop head parameters from state dict
        if drop_head:
            state_dict = {name: param for name, param in state_dict.items() if not name.startswith('head')}

        # Now load the remaining parameters
        self.load_state_dict(state_dict, strict=False)

        # freeze the backbone network layers
        if freeze_backbone:
            for name, param in self.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False


class GANModule(pl.LightningModule):
    def __init__(self, network_hparams, optimizer_hparams, loss_hparams, scheduler_hparams, metric_hparams,
                 input_shape, latent_dim, batch_size):
        super(GANModule, self).__init__()

        # ---------- backbone model ---------- #
        self.generator = ssrllib.util.common.create_module(network_hparams['generator'])
        self.discriminator = ssrllib.util.common.create_module(network_hparams['discriminator'])

        # ---------- input shape ---------- #
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1,) + tuple(input_shape), dtype=torch.float32)

        # ---------- optimizer, loss & metric ---------- #
        # Save optimizer params
        self.optimizer_hparams = optimizer_hparams
        self.optimizer_hparams['params'] = self.parameters()

        # Save scheduler params
        self.scheduler_hparams = scheduler_hparams

        # Create loss function
        self.loss_module = ssrllib.util.common.create_module(loss_hparams)

        # Create metric
        self.metric_name = metric_hparams['name']
        self.metric = ssrllib.util.common.create_module(metric_hparams)

        # ---------- latent space ---------- #
        self.n_sample = 10
        self.latent_dim = latent_dim

    def forward(self, x):
        sample_z = torch.randn(self.n_sample, self.latent)
        real_img = x
