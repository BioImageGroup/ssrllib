"""
This file contains the different heads to be attached to the HEADLESS - backbone for the different tasks
"""

import numpy as np
from torch import nn
from torch.nn import Linear, Sequential, Sigmoid, Tanh

from ssrllib.models.blocks import Deconv2D, AEDecoder


class LinearHead(nn.Module):
    def __init__(self, in_features, task_classes):
        super().__init__()

        self.linear = Linear(in_features=in_features, out_features=task_classes)

    def forward(self, x):
        return self.linear(x)


class AEDecoder2D(AEDecoder):

    def __init__(self, out_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                         dropout=dropout, activation=activation)

        for i in range(levels - 1):
            in_features = 2 ** (levels - i - 1) * feature_maps
            out_features = 2 ** (levels - i - 2) * feature_maps
            deconv = Deconv2D(in_channels=in_features, out_channels=out_features)
            self.features.add_module('deconv%d' % (i + 1), deconv)

        i += 1
        in_features = out_features
        out_features = out_channels
        deconv = Deconv2D(in_channels=in_features, out_channels=out_features)
        self.features.add_module('deconv%d' % (i + 1), deconv)

        self.output = nn.Tanh()

    def forward(self, inputs):
        outputs = inputs

        for i in range(self.levels):
            outputs = getattr(self.features, 'deconv%d' % (i + 1))(outputs)

        outputs = self.output(outputs)

        return outputs


class DecoderHead(nn.Module):
    def __init__(self, levels: int, in_features: int, task_classes: int):
        super().__init__()

        self.levels = levels
        self.in_features = in_features
        self.task_classes = task_classes

        self.decoder = Sequential()
        self._create_network()

    def _create_network(self):
        out_features = self.in_features

        for i in range(self.levels):
            in_features = out_features
            out_features = int(in_features/2)

            # assert out_features > 3, f'Too many levels for the current bottleneck size (output features: ' \
            #                          f'{out_features}), please increase the number of the last feature map of the ' \
            #                          f'backbone, or decrease the levels of the head'

            deconv = Deconv2D(in_channels=in_features, out_channels=out_features)
            self.decoder.add_module('deconv%d' % (i + 1), deconv)

        in_features = out_features
        out_features = self.task_classes
        deconv = Deconv2D(in_channels=in_features, out_channels=out_features)
        self.decoder.add_module('deconv%d' % (i + 2), deconv)
        self.decoder.add_module('out', Tanh())

    def forward(self, inputs):

        outputs = inputs
        outputs = outputs.view(outputs.shape[0], outputs.shape[1], 1, 1)

        for i in range(self.levels + 1):
            outputs = getattr(self.decoder, 'deconv%d' % (i + 1))(outputs)
        outputs = getattr(self.decoder, 'out')(outputs)
        return outputs


class ResnetDecoder(nn.Module):

    """
    This is the decoder part of the AutoEncoder. It takes in the code from the Encoder, and outputs an image
    """
    def __init__(self, latent_dim, channel_factor=32):
        super(ResnetDecoder, self).__init__()
        self.channels = 3

        self.conv1 = nn.ConvTranspose2d(latent_dim, channel_factor * 8, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channel_factor * 8)
        self.relu1 = nn.ReLU(True)
        # # out_shape: channel_factor * 8, 4, 4
        self.conv2 = nn.ConvTranspose2d(channel_factor * 8, channel_factor * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channel_factor * 4)
        self.relu2 = nn.ReLU(True)
        # out_shape: channel_factor * 4, 8, 8
        self.conv3 = nn.ConvTranspose2d(channel_factor * 4, channel_factor * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channel_factor * 2)
        self.relu3 = nn.ReLU(True)
        # out_shape: channel_factor * 2, 16, 16
        self.conv4 = nn.ConvTranspose2d(channel_factor * 2, channel_factor, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(channel_factor)
        self.relu4 = nn.ReLU(True)
        # out_shape: channel_factor, 32, 32
        self.conv5 = nn.ConvTranspose2d(channel_factor, self.channels, 4, 2, 1, bias=False)
        self.activ = Sigmoid()
        # out_shape: channels, 64, 64

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # print('asdasasd', x.shape)
        # x = x.view(x.size(0), 512, 7, 7)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.activ(x)
        return x
