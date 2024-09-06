from .basic_callback import Callback
import sys, os
import fastcore as fc

sys.path.append(os.path.abspath('..'))
from utils import to_cpu
from learners import Learner


class CapturePredsCB(Callback):
    def before_fit(self, learn): self.all_preds, self.all_targs = [], []

    def after_batch(self, learn):
        self.all_preds.append(to_cpu(learn.preds))
        self.all_targs.append(to_cpu(learn.batch[1]))


@fc.patch
def capture_preds(self: Learner, cbs=None, inps=False):
    cp = CapturePredsCB()
    self.fit(1, train=False, cbs=[cp] + fc.L(cbs))
    res = cp.all_preds, cp.all_targs
    if inps: res = res + (cp.all_inps,)
    return res