import torch.nn as nn


def conv(ni, nf, ks=3, stride=2, act=nn.ReLU, norm=None, bias=False):
    layers = [nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks // 2, bias=bias)]
    if norm: layers.append(norm(nf))
    if act: layers.append(act())
    return nn.Sequential(*layers)