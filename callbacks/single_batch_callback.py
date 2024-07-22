from .basic_callback import Callback
from .exceptions import CancelEpochException


class SingleBatchCB(Callback):
    order = 1
    def after_batch(self): raise CancelEpochException()