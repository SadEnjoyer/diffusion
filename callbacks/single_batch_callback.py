from .basic_callback import *
from .exceptions import *

class SingleBatchCB(Callback):
    order = 1
    def after_batch(self): raise CancelEpochException()