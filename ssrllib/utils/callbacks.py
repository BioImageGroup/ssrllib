from typing import Dict, List, Optional, Sequence, Union

from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning import LightningModule, Trainer, Callback
import requests
import torchvision

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from ssrllib.utils import io
import pytorch_lightning as pl


class ValidationVisualization(Callback):
    def __init__(self):
        pass

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        # x, y = batch
        # y_pred = outputs

        # num_rows = self.steps
        # images = []
        # for cls in torch.unique(y):
        #     images.append(x[y == cls][0])
        #     images.append(y_pred[y == cls][0])

        grid = torchvision.utils.make_grid(
            pl_module.retained, nrow=len(pl_module.retained)//2)
        str_title = f"val/reconstructions"
        trainer.logger.experiment.add_image(
            str_title, grid, global_step=trainer.global_step)


class TelegramBotInterfacer(Callback):
    def __init__(self, token: str, chat_ids: List[str], training_id: Dict[str, Union[str, int, float]], epoch_freq: int = 1):
        self.token = token
        self.chat_ids = chat_ids
        self.training_id = training_id
        self.epoch_freq = epoch_freq

        self.msg_url = "https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}"
        self.typing_url = "https://api.telegram.org/bot{}/sendChatAction?chat_id={}&action={}"
        self.training_title = self._get_train_id_as_string()
        self.send_msg("==============", keep_typing=True)
        self.send_msg("==============", keep_typing=True)
        self.send_msg(self.training_title, keep_typing=True)
        self.text_acc = ''

    def _get_train_id_as_string(self):
        task = self.training_id['dataset'].split('.')[-1].strip('Dataset').strip('ROI')
        data = self.training_id['data_dir'].split('/')[-1].strip('_pngseq')
        print('perc data: ', self.training_id['perc_data'])
        supervision = int((0.6 / self.training_id['perc_data']))

        return f'{task} training for {data.upper()} with {supervision}% of labels'

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        text = f'Starting {stage}'
        self.send_msg(text, keep_typing=True)

    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: Optional[str] = None) -> None:
        self.send_msg(f'Finished {stage}')

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._log_on_epoch_end(trainer, pl_module, stage='train')

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._log_on_epoch_end(trainer, pl_module, stage='val')
        # Send message only each epoch_freq epochs

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = trainer.logged_metrics

        current_metric = metrics["test/metric_epoch"]
        text = f'Test metric: {current_metric:.4f}'
        self.send_msg(text)

    def _log_on_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        metrics = trainer.logged_metrics
        current_metric = metrics[f"{stage}/metric_epoch"]

        if stage == 'train':
            self.text_acc += f'Epoch {trainer.current_epoch}'

        self.text_acc += f'\n{stage.capitalize()}: {current_metric:.4f}'

        # If validation include best model
        if stage == 'val':
            best_metric = trainer.checkpoint_callback.best_model_score  # could be None
            best_metric = float(best_metric) if best_metric else current_metric
            self.text_acc += f'(best: {best_metric:.4f})'

            # If validation and each epoch_freq epochs -> send message
            # if (trainer.current_epoch + 1) % self.epoch_freq == 0:
            self.send_msg(self.text_acc, keep_typing=True)
            self.text_acc = ''

    def send_msg(self, text: str, keep_typing: bool = False):
        for id in self.chat_ids:
            url = self.msg_url.format(self.token, id, text)
            results = requests.get(url)
        if keep_typing:
            self.set_typing()

    def set_typing(self, is_typing: str = 'typing'):
        for id in self.chat_ids:
            url = self.typing_url.format(self.token, id, is_typing)
            results = requests.get(url)
