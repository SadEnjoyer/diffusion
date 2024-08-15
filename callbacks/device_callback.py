from .basic_callback import Callback
import fastcore.all as fc
import sys
import os

sys.path.append(os.path.abspath('..'))
from utils import to_device


class DeviceCB(Callback):
    def __init__(self, device='cpu'): fc.store_attr()
    def before_fit(self, learn): learn.model.to(self.device)
    def before_batch(self, learn): learn.batch = to_device(learn.batch, device=self.device)
