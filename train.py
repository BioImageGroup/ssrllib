"""
    This is a script that illustrates training a 2D U-Net
"""
import argparse
import copy
import os
from multiprocessing import freeze_support

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.utilities.seed import seed_everything
from ssrllib.data.datamodule import DataModule
from ssrllib.util.common import create_module
from ssrllib.util.io import print_ts
from torchvision.transforms import Compose

if __name__ == '__main__':
    freeze_support()

    # ---------- PARAMETERS PARSING ---------- #
    # Parsing arguments
    print_ts('Parsing command-line arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the config file", type=str, required=True)
    args = parser.parse_args()
    config_name = args.config.split('/')[-1].split('.')[0]

    # Loading parameters parameters from yaml config file
    print_ts(f"Loading parameters parameters from {args.config} config file")
    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    params_to_save = copy.deepcopy(params)


    # ---------- DATA LOADING ---------- #
    # Seed all random processes
    print_ts(f"Seeding all stochastic processes with seed {params['seed']}")
    seed_everything(params['seed'])

    # Loading data from filesystem
    print_ts("Initializing datasets")
    transforms = []
    if 'transforms' in params:
        for t in params['transforms']:
            transforms.append(create_module(t))

    params['datamodule']['dataset_hparams']['train']['transforms'] = Compose(transforms)
    dm = DataModule(**params['datamodule'])


    # ---------- DOWNSTREAM MODEL LOADING ---------- #
    print_ts("Initializing neural network")
    net = create_module(params['model'])

    # load pretraned weights
    if 'pretrained' in params:
        net.load_from_pretext(**params['pretrained'])


    # ---------- CALLBACKS ---------- #
    callbacks = []
    if 'callbacks' in params:
        for callback_params in params['callbacks']:
            callbacks.append(create_module(callback_params))
        params['trainer']['callbacks'] = callbacks


    # ---------- FIT ---------- #
    # add root dir with the same name as the config file
    params['trainer']['default_root_dir'] = os.path.join('logs', config_name)
    trainer = pl.Trainer(**params['trainer'])

    # manually save the config file, we need to manually create the logdir because it is not
    # yet created at this point, but we still need to log the config file as soon as possible
    os.makedirs(trainer.log_dir, exist_ok=True)
    print_ts("Saving running configuration")
    with open(os.path.join(trainer.log_dir, 'config.yaml'), 'w') as f_out:
        yaml.dump(params_to_save, f_out)

    # start the actual training
    trainer.fit(net, datamodule=dm)


    # ---------- TEST ---------- #
    print_ts('Validating the network')
    # load the best network (on validation metric) and then run a test loop
    net.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    dm.setup(stage="test")
    trainer.test(net, datamodule=dm)
