import random, torch.nn as nn


def _rand_copy1(x, pct):
    szx = int(pct * x.shape[-2])
    szy = int(pct * x.shape[-1])
    stx1 = int(random.random() * (1 - pct) * x.shape[-2])
    sty1 = int(random.random() * (1 - pct) * x.shape[-1])
    stx2 = int(random.random() * (1 - pct) * x.shape[-2])
    sty2 = int(random.random() * (1 - pct) * x.shape[-1])
    x[:, :, stx1:stx1 + szx, sty1:sty1 + szy] = x[:, :, stx2:stx2 + szx, sty2:sty2 + szy]


def random_copy(x, pct=0.2, max_num=4):
    num = random.randint(0, max_num)
    for _ in range(num): _rand_copy1(x, pct)
    return x


class RandCopy(nn.Module):
    def __init__(self, pct=0.2, max_num=4):
        super().__init__()
        self.pct, self.max_num = pct, max_num

    def forward(self, x): return random_copy(x, self.pct, self.max_num)