from .basic_callback import Callback


class BatchTransformCB(Callback):
    def __init__(self, tfm): self.tfm = tfm
    def before_batch(self): self.learn.batch = self.tfm(self.learn.batch)