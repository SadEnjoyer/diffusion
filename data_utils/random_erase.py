import random, torch.nn as nn
from torch.nn import init


def _rand_erase1(x, pct, xm, xs, mn, mx):
    szx = int(pct * x.shape[-2])
    szy = int(pct * x.shape[-1])
    stx = int(random.random() * (1 - pct) * x.shape[-2])
    sty = int(random.random() * (1 - pct) * x.shape[-1])
    init.normal_(x[:, :, stx:stx + szx, sty:sty + szy], mean=xm, std=xs)
    x.clamp_(mn, mx)


def rand_erase(x, pct=0.2, max_num=4):
    xm, xs, mn, mx = x.mean(), x.std(), x.min(), x.max()
    num = random.randint(0, max_num)
    for _ in range(num): _rand_erase1(x, pct, xm, xs, mn, mx)
    return x


class RandErase(nn.Module):
    def __init__(self, pct=0.2, max_num=4):
        super().__init__()
        self.pct, self.max_num = pct, max_num

    def forward(self, x): return rand_erase(x, self.pct, self.max_num)