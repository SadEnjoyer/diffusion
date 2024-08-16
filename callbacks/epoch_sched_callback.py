from .base_sched_callback import BaseSchedCB


class EpochSchedCB(BaseSchedCB):
    def after_epoch(self, learn): self.step(learn)