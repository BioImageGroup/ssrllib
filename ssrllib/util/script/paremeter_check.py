import argparse
from pprint import pprint

import yaml

if __name__ == '__main__':

    # Parsing arguments
    print('Parsing command-line arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Path to the configuration file", type=str, default="meso_autoencoding.yaml")
    args = parser.parse_args()

    # Loading parameters parameters from yaml config file
    print(f"Loading parameters parameters from {args.config} config file")
    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    pprint(params)

    print((1,) + tuple(params['downstream_model']['input_shape']))