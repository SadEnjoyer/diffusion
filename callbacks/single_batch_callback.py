from .basic_callback import Callback
from .exceptions import CancelFitException


class SingleBatchCB(Callback):
    order = 1
    def after_batch(self, learn): raise CancelFitException()