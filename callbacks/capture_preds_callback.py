from .basic_callback import Callback
import sys, os

sys.path.append(os.path.abspath('..'))
from utils import to_cpu


class CapturePredsCB(Callback):
    def before_fit(self, learn): self.all_preds, self.all_targs = [], []

    def after_batch(self, learn):
        self.all_preds.append(to_cpu(learn.preds))
        self.all_targs.append(to_cpu(learn.batch[1]))