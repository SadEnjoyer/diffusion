import torch, torch.nn as nn
from .device_callback import DeviceCB


class MultiGPUsCallback(DeviceCB):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def before_fit(self, learn):
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for training")
            learn.model = nn.DataParallel(learn.model)
        super().before_fit(learn)

    def after_fit(self, learn):
        if isinstance(learn.model, nn.DataParallel):
            learn.model = learn.model.module
