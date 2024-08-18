import torch, fastcore.all as fc
from .basic_callback import Callback
from .device_callback import DeviceCB


def noisify(x_0, alpha_bar, n_steps=1000):
    device = x_0.device
    n = len(x_0)
    t = torch.randint(0, n_steps, (n,), dtype=torch.long)
    epsilon = torch.randn(x_0.shape, device=device)
    alpha_bar_t = alpha_bar[t].reshape(-1, 1, 1, 1).to(device)
    xt = alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * epsilon
    return xt, t.to(device), epsilon


@torch.no_grad()
def sample(model, sz, alpha, alphabar, sigma, n_steps):
    device = next(model.parameters()).device
    x_t = torch.randn(sz, device=device)
    preds = []
    for t in reversed(range(n_steps)):
        t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
        z = (torch.randn(x_t.shape) if t > 0 else torch.zeros(x_t.shape)).to(device)
        alpha_bar_t1 = alphabar[t - 1] if t > 0 else torch.tensor(1)
        beta_bar_t = 1 - alphabar[t]
        beta_bar_t1 = 1 - alpha_bar_t1
        x_0_hat = ((x_t - beta_bar_t.sqrt() * model((x_t, t_batch))) / alphabar[t].sqrt()).clamp(-1, 1)
        x_t = x_0_hat * alpha_bar_t1.sqrt() * (1 - alpha[t]) / beta_bar_t + \
            x_t * alpha[t].sqrt() * beta_bar_t1 / beta_bar_t + sigma[t] * z
        preds.append(x_t.cpu())
    return preds


class DDPMCB(Callback):
    order = DeviceCB.order + 1

    def __init__(self, n_steps, beta_min, beta_max):
        super().__init__()
        fc.store_attr()
        self.beta = torch.linspace(self.beta_min, self.beta_max, self.n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma = self.beta.sqrt()

    def before_batch(self, learn): learn.batch = noisify(learn.batch[0], self.alpha_bar)
    def sample(self, model, sz): return sample(model, sz, self.alpha, self.alpha_bar, self.sigma, self.n_steps)


class DDPMCB2(Callback):
    def after_predict(self, learn): learn.preds = learn.preds.sample