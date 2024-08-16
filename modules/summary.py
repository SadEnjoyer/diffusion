import fastcore.all as fc, os, sys

sys.path.append(os.path.abspath('..'))

from learners import Learner
from utils import Hooks
from callbacks import SingleBatchCB


@fc.patch
def summary(self: Learner):
    res = '|Module|Input|Output|Num params|\n|--|--|--|--|\n'
    tot = 0

    def _f(hook, mod, inp, outp):
        nonlocal res, tot
        nparms = sum(o.numel() for o in mod.parameters())
        tot += nparms
        res += f'|{type(mod).__name__}|{tuple(inp[0].shape)}|{tuple(outp.shape)}|{nparms}|\n'
    try:
        with Hooks(self.model, _f) as hooks: self.fit(1, lr=1, cbs=SingleBatchCB())
    except: pass
    print("Tot params: ", tot)
    if fc.IN_NOTEBOOK:
        from IPython.display import Markdown
        return Markdown(res)
    else:
        print(res)