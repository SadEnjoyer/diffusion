from .hooks_callback import HooksCallback
import numpy as np
import matplotlib.pyplot as plt
import fastcore.all as fc
import sys
import os

sys.path.append(os.path.abspath('..'))

from utils import Hooks, append_stats
from utils.plots import get_grid, show_image
from utils.append_stats import get_hist, get_min


class ActivationStats(HooksCallback):
    def __init__(self, mod_filter=fc.noop): super().__init__(append_stats, mod_filter)

    def color_dim(self, figsize=(11, 5)):
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin='lower')

    def dead_chart(self, figsize=(11, 5)):
        fig, axes = get_grid(len(self), figsize=figsize)
        for ax, h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0, 1)

    def plot_stats(self, figsize=(10, 4)):
        fig, axs = plt.subplots(1, 2, figsize=figsize)
        for h in self:
            for i in 0, 1: axs[i].plot(h.stats[i])
        axs[0].set_title('Means')
        axs[1].set_title("Stdevs")
        plt.legend(fc.L.range(self))
