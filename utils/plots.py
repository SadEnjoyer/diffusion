import math, fastcore.all as fc, matplotlib.pyplot as plt, numpy as np, sys, os
from itertools import zip_longest
from .append_stats import *

sys.path.append(os.path.abspath('..'))
from learners import Learner
from callbacks import SingleBatchCB


@fc.delegates(plt.Axes.imshow)
def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if fc.hasattrs(im, ('cpu', 'permute', 'detach')):
        im = im.detach().cpu()
        if len(im.shape) == 3 and im.shape[0] < 5: im = im.permute(1, 2, 0)
    elif not isinstance(im, np.ndarray): im = np.array(im)
    if im.shape[-1] == 1: im = im[..., 0]
    if ax is None: _, ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if noframe: ax.axis('off')
    return ax


@fc.delegates(plt.subplots, keep=True)
def subplots(
    nrows: int = 1,  # Number of rows in returned axes grid
    ncols: int = 1,  # Number of columns in returned axes grid
    figsize: tuple = None,  # Width, height in inches of the returned figure
    imsize: int = 3,  # Size (in inches) of images that will be displayed in the returned figure
    suptitle: str = None,  # Title to be set to returned figure
    **kwargs
):  # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None: figsize = (ncols * imsize, nrows * imsize)
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None: fig.suptitle(suptitle)
    if nrows * ncols == 1: ax = np.array([ax])
    return fig, ax


@fc.delegates(subplots)
def get_grid(
    n: int,  # Number of axes
    nrows: int = None,  # Number of rows, defaulting to `int(math.sqrt(n))`
    ncols: int = None,  # Number of columns, defaulting to `ceil(n/rows)`
    title: str = None,  # If passed, title set to the figure
    weight: str = 'bold',  # Title font weight
    size: int = 14,  # Title font size
    **kwargs,
):  # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows: ncols = ncols or int(np.floor(n / nrows))
    elif ncols: nrows = nrows or int(np.ceil(n / ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n / nrows))
    fig, axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows * ncols): axs.flat[i].set_axis_off()
    if title is not None: fig.suptitle(title, weight=weight, size=size)
    return fig, axs


@fc.delegates(subplots)
def show_images(ims: list,  # Images to show
                nrows: int | None = None,  # Number of rows in grid
                ncols: int | None = None,  # Number of columns in grid (auto-calculated if None)
                titles: list | None = None,  # Optional list of titles for each image
                **kwargs):
    "Show all images `ims` as subplots with `rows` using `titles`"
    axs = get_grid(len(ims), nrows, ncols, **kwargs)[1].flat
    for im, t, ax in zip_longest(ims, titles or [], axs): show_image(im, ax=ax, title=t)


@fc.patch
@fc.delegates(show_images)
def show_image_batch(self: Learner, max_n=9, cbs=None, **kwargs):
    self.fit(1, cbs=[SingleBatchCB()] + fc.L(cbs))
    show_images(self.batch[0][:max_n], **kwargs)
