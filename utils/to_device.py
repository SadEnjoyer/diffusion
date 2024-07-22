import torch
from collections.abc import Mapping


def to_device(x, device='cuda:0' if torch.cuda.is_available() else 'cpu'):
    if isinstance(x, torch.Tensor): return x.to(device)
    if isinstance(x, Mapping): return {k: v.to(device) for k, v in x.items()}
    return x


def to_cpu(x):
    if isinstance(x, Mapping): return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    return x.detach().cpu()