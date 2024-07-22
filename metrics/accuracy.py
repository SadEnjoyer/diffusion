from .metric import Metric


class Accuracy(Metric):
    def calc(self, inps, targs): return (inps == targs).float().mean()