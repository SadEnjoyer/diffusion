import fastcore.all as fc, sys, os

sys.path.append(os.path.abspath('..'))

from learners import TrainLearner
from utils import Hooks
from callbacks import SingleBatchCB


def _flops(x, h, w):
    if x.dim() < 3: return x.numel()
    if x.dim() == 4: return x.numel() * h * w


@fc.patch
def summary(self: TrainLearner):
    res = '|Module|Input|Output|Num params|MFLOPS|\n|--|--|--|--|--|\n'
    totp, totf = 0, 0

    def _f(hook, mod, inp, outp):
        nonlocal res, totp, totf
        nparms = sum(o.numel() for o in mod.parameters())
        totp += nparms
        *_, h, w = outp.shape
        flops = sum(_flops(o, h, w) for o in mod.parameters()) / 1e6
        totf += flops
        res += f'|{type(mod).__name__}|{tuple(inp[0].shape)}|{tuple(outp.shape)}|{nparms}|{flops:.1f}|\n'
    try:
        with Hooks(self.model, _f) as hooks: self.fit(1, lr=1, cbs=SingleBatchCB())
    except:
        pass
    print(f"Tot params: {totp}; MFLOPS: {totf:.1f}")
    if fc.IN_NOTEBOOK:
        from IPython.display import Markdown
        return Markdown(res)
    else: print(res)