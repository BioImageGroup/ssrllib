"""
    This is a script that illustrates training a 2D U-Net
"""
from re import A
import numpy as np
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

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, NMF
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import classification_report

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

    # params['datamodule']['dataset_hparams']['train']['transforms'] = Compose(transforms)
    dm = DataModule(**params['datamodule'])


    # ---------- DOWNSTREAM MODEL LOADING ---------- #
    print_ts("Initializing neural network")
    net = create_module(params['model'])

    # load pretraned weights
    if 'pretrained' in params:
        net.load_from_pretext(**params['pretrained'])

    # ---------- FEATURE EXTRACTION ---------- #
    # add root dir with the same name as the config file
    params['trainer']['default_root_dir'] = os.path.join('logs', config_name)
    trainer = pl.Trainer(**params['trainer'])

    # manually save the config file, we need to manually create the logdir because it is not
    # yet created at this point, but we still need to log the config file as soon as possible
    os.makedirs(trainer.log_dir, exist_ok=True)
    print_ts("Saving running configuration")
    with open(os.path.join(trainer.log_dir, 'config.yaml'), 'w') as f_out:
        yaml.dump(params_to_save, f_out)


    # use the pretrained network to perform feature extraction
    out = trainer.predict(net, datamodule=dm)

    features = []
    labels = []
    for elem in out:
        feature, label = elem
        features.append(np.array(feature[0]))
        labels.append(int(label))

    # we alwayws want the same test set
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=params['pred']['test_size'], shuffle=True, random_state=0)
    
    # and then we discard some of the data if we want to test for lower levels of supervision
    if params['pred']['train_size'] != 1:
        x_train, _, y_train, _ = train_test_split(x_train, y_train, train_size=params['pred']['train_size'], shuffle=True, random_state=0)

    print_ts(f'test class distribution: {np.bincount(y_test)}')

    downsample_enn = EditedNearestNeighbours(sampling_strategy='majority')
    upsample_smote = SMOTE(random_state=0)
    svc = SVC()

    pipeline = Pipeline([
        ('downsampler', downsample_enn), 
        ('upsampler', upsample_smote),
        ('classifier', svc),
    ])

    grid = GridSearchCV(pipeline, cv=5, n_jobs=4, param_grid=params['param_grid'], scoring='accuracy')

    grid.fit(x_train, y_train)
    print_ts(f'Grid search found the best parameter config: \n{grid.best_params_}\n and score {grid.best_score_}')
    y_pred = grid.predict(x_test)
    
    print_ts(f"Prediction using a Linear SVM over {params['pred']['train_size']*100}% of labelled data")
    print_ts(f'Classification report:\n{classification_report(y_test, y_pred)}')

    report = classification_report(y_test, y_pred, output_dict=True)

    # manually save the config file, we need to manually create the logdir because it is not
    # yet created at this point, but we still need to log the config file as soon as possible
    os.makedirs(trainer.log_dir, exist_ok=True)
    print_ts(f"Saving output prediction at {trainer.log_dir}/prediction_resluts.yaml")
    with open(os.path.join(trainer.log_dir, 'prediction_resluts.yaml'), 'w') as f_out:
        yaml.dump(report, f_out)