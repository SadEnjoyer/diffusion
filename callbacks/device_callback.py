from .basic_callback import *
import fastcore.all as fc
import sys
import os

sys.path.append(os.path.abspath('..'))
from utils import *

class DeviceCB(Callback):
    def __init__(self, device='cpu'): fc.store_attr()
    def before_fit(self): self.learn.model.to(self.device)
    def before_batch(self): self.learn.batch = to_device(self.learn.batch, device=self.device)
