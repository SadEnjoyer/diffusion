import torch, sys, os
sys.path.append(os.path.abspath('..'))
from utils.to_device import to_cpu
from utils.hook import Hook


def _lsuv_stats(hook, mod, inp, outp):
    acts = to_cpu(outp)
    hook.mean = acts.mean()
    hook.std = acts.std()


def lsuv_init(m, m_in, xb, model):
    h = Hook(m, _lsuv_stats)
    with torch.no_grad():
        while model(xb) is not None and (abs(h.std - 1) > 1e-3 or abs(h.mean) > 1e-3):
            m_in.bias -= h.mean
            m_in.weight.data /= h.std
    h.remove()