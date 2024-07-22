from .basic_callback import Callback
import math
from .exceptions import CancelBatchException, \
    CancelEpochException, CancelFitException
import fastcore.all as fc
import matplotlib.pyplot as plt

import sys
import os

sys.path.append(os.path.abspath('..'))

from utils import to_cpu


class LRFinderCB(Callback):
    """
    https://arxiv.org/pdf/1506.01186
    """
    def __init__(self, lr_mult=1.3):
        fc.store_attr()

    def before_fit(self):
        self.lrs, self.losses = [], []
        self.min = math.inf

    def after_batch(self):
        if not self.learn.model.training: raise CancelEpochException()

        self.lrs.append(self.learn.opt.param_groups[0]['lr'])
        loss = to_cpu(self.learn.loss)
        self.losses.append(loss)

        if loss < self.min: self.min = loss
        if loss > self.min * 3: raise CancelFitException()
        for g in self.learn.opt.param_groups: g['lr'] *= self.lr_mult

    def after_fit(self):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
