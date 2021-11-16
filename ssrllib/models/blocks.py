import math
from typing import Optional, Callable

from torch import nn, Tensor


################################################
# ------------------ RESNET ------------------ #
################################################

class PreActResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample:
            c_out = c_in

        # Network representing F
        self.net = nn.Sequential(
            nn.BatchNorm2d(c_in),
            act_fn(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1 if not subsample else 2, bias=False),
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
        )

        # 1x1 convolution needs to apply non-linearity as well as not done on skip connection
        self.downsample = (
            nn.Sequential(nn.BatchNorm2d(c_in), act_fn(), nn.Conv2d(c_in, c_out, kernel_size=1, stride=2, bias=False))
            if subsample
            else None
        )

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out


# From the official pytorch implementation
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


#################################################
# ---------------- AUTOENCODER ---------------- #
#################################################
class Conv2D(nn.Module):
    """
    2D convolutional layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional kernel_size: kernel size of the convolutions
    :param optional stride: stride of the convolutions
    :param optional padding: padding used for border cases ("SAME", "VALID" or None)
    :param optional bias: use bias term or not
    :param optional dilation: dilation of the convolution
    :param optional dropout: dropout factor
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    :param optional norm: specify normalization ("batch", "instance" or None)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='SAME', bias=True, dilation=1,
                 dropout=0.0, activation=None, norm=None):
        super().__init__()

        if padding == 'SAME':
            p = kernel_size // 2
        else:  # VALID (no) padding
            p = 0

        # initialize convolutional block
        self.unit = nn.Sequential(nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation))

        # apply normalization
        if norm == 'batch':
            self.unit.add_module('norm', nn.BatchNorm2d(int(out_channels)))
        elif norm == 'instance':
            self.unit.add_module('norm', nn.InstanceNorm2d(int(out_channels)))

        # apply dropout
        if dropout > 0.0:
            self.unit.add_module('dropout', nn.Dropout2d(p=dropout))

        # apply activation
        if activation == 'relu':
            self.unit.add_module('activation', nn.ReLU())
        elif activation == 'sigmoid':
            self.unit.add_module('activation', nn.Sigmoid())

    def forward(self, inputs):

        return self.unit(inputs)


class Deconv2D(nn.Module):
    """
    2D de-convolutional layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param optional deconv: us deconvolution or upsampling layers
    :param optional bias: use bias term or not
    :param optional activation: specify activation function ("relu", "sigmoid" or None)
    """

    def __init__(self, in_channels, out_channels, bias=True, activation='relu'):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

    def forward(self, inputs):
        return self.activation(self.up(inputs))


class AEEncoder(nn.Module):
    """
    AutoEncoder encoder base class
    """

    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__()

        self.in_channels = in_channels
        self.features = nn.Sequential()
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation


class AEDecoder(nn.Module):
    """
    AutoEncoder decoder base class
    """

    def __init__(self, out_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__()

        self.out_channels = out_channels
        self.features = nn.Sequential()
        self.feature_maps = feature_maps
        self.levels = levels
        self.norm = norm
        self.dropout = dropout
        self.activation = activation


class AEEncoder2D(AEEncoder):
    def __init__(self, in_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(in_channels=in_channels, feature_maps=feature_maps, levels=levels, norm=norm, dropout=dropout,
                         activation=activation)

        in_features = in_channels
        for i in range(levels):
            out_features = (2 ** i) * feature_maps
            # print(f'encoder level {i} out features {in_features} -> {out_features}')

            # convolutional block
            conv_block = Conv2D(in_features, out_features, norm=norm, dropout=dropout, activation=activation)
            self.features.add_module('convblock%d' % (i + 1), conv_block)

            # pooling
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.features.add_module('pool%d' % (i + 1), pool)

            # input features for next block
            in_features = out_features

    def forward(self, inputs):
        outputs = inputs

        for i in range(self.levels):
            outputs = getattr(self.features, 'convblock%d' % (i + 1))(outputs)
            outputs = getattr(self.features, 'pool%d' % (i + 1))(outputs)
            # print(f'Encoder level {i} output shape: {outputs.shape}')

        return outputs


class AEDecoder2D(AEDecoder):

    def __init__(self, out_channels=1, feature_maps=64, levels=4, norm='instance', dropout=0.0, activation='relu'):
        super().__init__(out_channels=out_channels, feature_maps=feature_maps, levels=levels, norm=norm,
                         dropout=dropout,
                         activation=activation)

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

