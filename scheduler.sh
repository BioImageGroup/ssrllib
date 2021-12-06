#!/usr/bin/zsh

python predictor.py --config /home/nicola/ssrllib/configs/crc/autoencoding/downstream/crc_autoencoding_downstream.yaml
python predictor.py --config /home/nicola/ssrllib/configs/crc/autoencoding/downstream/crc_autoencoding_downstream_metta.yaml

python predictor.py --config /home/nicola/ssrllib/configs/crc/rotation/downstream/crc_rotation_downstream.yaml
python predictor.py --config /home/nicola/ssrllib/configs/crc/rotation/downstream/crc_rotation_downstream_metta.yaml

python predictor.py --config /home/nicola/ssrllib/configs/crc/imagenet_pretrained/downstream/crc_imagenet_downstream.yaml
python predictor.py --config /home/nicola/ssrllib/configs/crc/imagenet_pretrained/downstream/crc_imagenet_downstream_metta.yaml

python predictor.py --config /home/nicola/ssrllib/configs/crc/jigsaw/downstream/crc_jigsaw_downstream.yaml
python predictor.py --config /home/nicola/ssrllib/configs/crc/jigsaw/downstream/crc_jigsaw_downstream_metta.yaml

