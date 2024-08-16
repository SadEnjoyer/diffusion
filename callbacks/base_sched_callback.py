from .basic_callback import Callback


class BaseSchedCB(Callback):
    def __init__(self, sched): self.sched = sched
    def before_fit(self, learn): self.schedo = self.sched(learn.opt)

    def step(self, learn):
        if learn.training: self.schedo.step()