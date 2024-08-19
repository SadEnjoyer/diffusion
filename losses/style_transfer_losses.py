from torch import tensor
import fastcore.all as fc, torch
from fastcore.foundation import L


def normalize(im, means=[0.485, 0.456, 0.406], stds=[0.229, 0.224, 0.225]):
    assert len(means) == len(stds) == 3
    imagenet_mean = tensor(means)[:, None, None].to(im.device)
    imagenet_std = tensor(stds)[:, None, None].to(im.device)
    return (im - imagenet_mean) / imagenet_std


def calc_features(imgs, target_layers=[18, 25], model=None):
    x = normalize(imgs)
    feats = []
    for i, layer in enumerate(model[:max(target_layers) + 1]):
        x = layer(x)
        if i in target_layers:
            feats.append(x.clone())
    return feats


class ContentLossToTarget():
    def __init__(self, target_im, target_layers=[18, 25]):
        fc.store_attr()
        with torch.no_grad():
            self.target_features = calc_features(target_im, target_layers)

    def __call__(self, input_im):
        return sum((f1 - f2).pow(2).mean() for f1, f2 in
                   zip(calc_features(input_im, self.target_layers),
                       self.target_features
                       ))


def calc_grams(img, target_layers=[1, 6, 11, 18, 25]):
    return L(torch.einsum('chw, dhw -> cd', x, x) / (x.shape[-2] * x.shape[-1])
             for x in calc_features(img, target_layers))


class StyleLossToTarget():
    def __init__(self, target_im, target_layers=(1, 6, 11, 18, 25)):
        fc.store_attr()
        with torch.no_grad(): self.target_grams = calc_grams(target_im, target_layers)

    def __call__(self, input_im):
        return sum((f1 - f2).pow(2).mean() for f1, f2 in
                   zip(calc_grams(input_im, self.target_layers), self.target_grams))