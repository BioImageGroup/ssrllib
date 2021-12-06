import os
import yaml
from pytorch_lightning.utilities.cli import LightningCLI
from ssrllib.utils.io import print_ts
from ssrllib.data import datamodules
from ssrllib.models import modules
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


if __name__ == "__main__":
    import pytorch_lightning as pl

    # run=false is not needed, but used for consistency
    # save_config_overwrite=True is needed to avoid errors when test overwrites fit config file
    cli = LightningCLI(run=False, save_config_overwrite=True)

    # Predict
    out = cli.trainer.predict(cli.model, cli.datamodule)

    # Extract embeddings and ground truth
    embeddings = []
    labels = []
    for elem in out:
        feature, label = elem
        embeddings.append(np.array(feature[0]))
        labels.append(int(label))

    # Split away test data
    x, x_test, y, y_test = train_test_split(
        embeddings, labels, test_size=cli.model.splits['test'][0], shuffle=True, random_state=0)
    print_ts(f'test class distribution: {np.bincount(y_test)}')

    # For every considered portion of training data
    for train_size in cli.model.splits['train']:

        print_ts(f"Predicting using a Linear SVM over {train_size*100}% of labelled data")
        # discard some of the data if we want to test for lower levels of supervision
        if train_size != 1:
            x_train, _, y_train, _ = train_test_split(
                x, y, train_size=train_size, shuffle=True, random_state=0)
        else:
            x_train = x
            y_train = y

        print_ts(f'train class distribution: {np.bincount(y_train)}')

        # fit the grid_search
        cli.model.gscv.fit(x_train, y_train)
        print_ts(f'Grid search found the best parameter config: '
                 f'\n{cli.model.gscv.best_params_}\n and score {cli.model.gscv.best_score_}')

        # get predictions
        y_pred = cli.model.gscv.predict(x_test)
        print_ts(f'Classification report:\n{classification_report(y_test, y_pred)}')

        report = classification_report(y_test, y_pred, output_dict=True)
        perc = int(train_size*100)

        os.makedirs(cli.trainer.log_dir, exist_ok=True)
        print_ts(f"Saving output prediction at {cli.trainer.log_dir}/prediction_resluts.yaml")
        with open(os.path.join(cli.trainer.log_dir, f'{perc}_resluts.yaml'), 'w') as f_out:
            yaml.dump(report, f_out)
