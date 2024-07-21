from .learner import *
from torch import tensor, nn, optim
import torch

class MomentumLearner(Learner):
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=optim.SGD, mom=0.85): 
        self.mom = mom
        super().__init__(model, dls, loss_func, lr, cbs, opt_func=optim.SGD)
    def predict(self): self.preds = self.model(self.batch[0])
    def get_loss(self): self.loss = self.loss_func(self.preds, self.batch[1])
    def backward(self): self.loss.backward()
    def step(self): self.opt.step()
    def zero_grad(self):
        with torch.no_grad():
            for p in self.model.parameters(): p.grad *= self.mom