import torch
import torch.nn as nn

# Some layers are from https://github.com/pytorch/vision/tree/master/torchvision/models

nonlinearity = nn.ReLU


def conv3x3(in_planes, out_planes, stride=1, padding=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=bias)


class Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nonlinearity(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride,
                             padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nonlinearity(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, is_deconv=True,
                 deflation=4, xavier=False):
        super(DecoderBlock, self).__init__()

        # B, C, H, W -> B, C/4, H, W
        assert in_channels % deflation == 0, "Incorrect deflation"
        self.conv1 = nn.Conv2d(in_channels, in_channels // deflation, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // deflation)
        self.relu1 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        if is_deconv:
            self.upscale = nn.ConvTranspose2d(in_channels // deflation, in_channels // deflation, 3,
                                              stride=2, padding=1, output_padding=1)
        else:
            self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.norm2 = nn.BatchNorm2d(in_channels // deflation)
        self.relu2 = nonlinearity(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // deflation, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity(inplace=True)

        if xavier:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.upscale(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2,
                 xavier=False):
        super(FPNBlock, self).__init__()

        mid_channels = int(round((in_channels + out_channels) / 2))
        self.conv1 = conv3x3(in_channels, mid_channels)
        self.relu1 = nonlinearity(inplace=True)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = conv3x3(mid_channels, out_channels)
        self.relu2 = nonlinearity(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if scale_factor > 1:
            self.upscale = nn.Upsample(scale_factor=scale_factor,
                                       mode='bilinear', align_corners=True)
        else:
            self.upscale = lambda x: x

        if xavier:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.upscale(x)
        return x
