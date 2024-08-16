from .base_sched_callback import BaseSchedCB


class BatchSchedCB(BaseSchedCB):
    def after_batch(self, learn): self.step(learn)