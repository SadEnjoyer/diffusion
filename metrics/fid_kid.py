import torch, fastcore as fc, sys, os, math
from torch import tensor
from scipy import linalg

sys.path.append(os.path.abspath('..'))
from learners import TrainLearner
from data_utils import DataLoaders


def _sqrt_newton_shulz(mat, num_iter=100):
    mat_norm = mat.norm()
    mat = mat.double()
    Y = mat / mat_norm
    n = len(mat)
    II = torch.eye(n, n).to(mat)
    Z = torch.eye(n, n).to(mat)

    for i in range(num_iter):
        T = (3 * II - Z @ Y) / 2
        Y, Z = Y @ T, T @ Z
        res = Y * mat_norm.sqrt()
        if ((mat - (res @ res)).norm() / mat_norm).abs() <= 1e-6: break
    return res


def _calc_stats(feats):
    feats = feats.squeeze()
    return feats.mean(0), feats.T.cov()


def _calc_fid(m1, c1, m2, c2):
    # csr = _sqrtm_newton_schulz(c1@c2)
    csr = tensor(linalg.sqrtm(c1 @ c2, 256).real)
    return (((m1 - m2)**2).sum() + c1.trace() + c2.trace() - 2 * csr.trace()).item()


def _squared_mmd(x, y):
    def k(a, b): return (a @ b.transpose(-2, -1) / a.shape[-1] + 1)**3
    m, n = x.shape[-2], y.shape[-2]
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    kxx_sum = kxx.sum([-1, -2]) - kxx.diagonal(0, -1, -2).sum(-1)
    kyy_sum = kyy.sum([-1, -2]) - kyy.diagonal(0, -1, -2).sum(-1)
    kxy_sum = kxy.sum([-1, -2])
    return kxx_sum / m / (m - 1) + kyy_sum / n / (n - 1) - kxy_sum * 2 / m / n


def _calc_kid(x, y, maxs=50):
    xs, ys = x.shape[0], y.shape[0]
    n = max(math.ceil(min(xs / maxs, ys / maxs)), 4)
    mmd = 0.
    for i in range(n):
        cur_x = x[round(i * xs / n):round((i + 1) * xs / n)]
        cur_y = y[round(i * ys / n):round((i + 1) * ys / n)]
        mmd += _squared_mmd(cur_x, cur_y)
    return (mmd / n).item()


class ImageEval:
    def __init__(self, model, dls, cbs=None):
        self.learn = TrainLearner(model, dls, loss_func=fc.noop, cbs=cbs, opt_func=None)
        self.feats = self.learn.capture_preds()[0].float().cpu().squeeze()
        self.stats = _calc_stats(self.feats)

    def get_feats(self, samp):
        self.learn.dls = DataLoaders([], [(samp, tensor([0]))])
        return self.learn.capture_preds()[0].float().cpu().squeeze()

    def fid(self, samp): return _calc_fid(*self.stats, *_calc_stats(self.get_feats(samp)))
    def kid(self, samp): return _calc_kid(self.feats, self.get_feats(samp))