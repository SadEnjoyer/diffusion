from .basic_callback import Callback
import fastcore.all as fc
import os, sys

sys.path.append(os.path.abspath('..'))

from utils import Hooks, append_stats


class HooksCallback(Callback):
    def __init__(self, hookfunc, mod_filter=fc.noop):
        fc.store_attr()
        super().__init__()

    def before_fit(self):
        mods = fc.filter_ex(self.learn.model.modules(), self.mod_filter)
        self.hooks = Hooks(mods, self._hookfunc)

    def _hookfunc(self, *args, **kwargs):
        if self.learn.model.training:
            self.hookfunc(*args, **kwargs)

    def after_fit(self): self.hooks.remove()
    def __iter__(self): return iter(self.hooks)
    def __len__(self): return len(self.hooks)
