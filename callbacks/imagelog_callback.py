from .basic_callback import Callback
from .train_callback import TrainCB
from .progress_callback import ProgressCB
from fastcore.all import store_attr
import sys, os

sys.path.append(os.path.abspath('..'))

from utils.plots import show_images
from utils.to_device import to_cpu


class ImageLogCB(Callback):
    order = ProgressCB.order + 1

    def __init__(self, log_every=10):
        store_attr()
        self.images = []
        self.i = 0

    def after_batch(self, learn):
        if self.i % self.log_every == 0: self.images.append(to_cpu(learn.preds.clip(0, 1)))
        self.i += 1

    def after_fit(self, learn): show_images(self.images)


class ImageOptCB(TrainCB):
    def predict(self, learn): learn.preds = learn.model()
    def get_loss(self, learn): learn.loss = learn.loss_func(learn.preds)