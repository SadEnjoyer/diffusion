from .train_learner import TrainLearner
from torch import tensor, nn, optim
import torch


class MomentumLearner(TrainLearner):
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=optim.SGD, mom=0.85): 
        self.mom = mom
        super().__init__(model, dls, loss_func, lr, cbs, opt_func=optim.SGD)

    def zero_grad(self):
        with torch.no_grad():
            for p in self.model.parameters(): p.grad *= self.mom