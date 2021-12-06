from pytorch_lightning.utilities.cli import LightningCLI, LightningArgumentParser
from sklearn.base import ClassifierMixin


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_class_arguments(ClassifierMixin, 'ClassifierMixin.init')
