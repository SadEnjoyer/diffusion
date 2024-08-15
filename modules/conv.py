import torch.nn as nn


def conv(input_channels, output_channels, ks=3, stride=2, act=nn.ReLU, norm=None, bias=False):
    layers = [nn.Conv2d(
        input_channels,
        output_channels,
        stride=stride,
        kernel_size=ks,
        padding=ks // 2,
        bias=bias)]
    if norm: layers.append(norm(output_channels))
    if act: layers.append(act())
    return nn.Sequential(*layers)