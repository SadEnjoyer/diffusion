from operator import itemgetter
from typing import Any, List
from collections.abc import Callable
from datasets import Dataset
from torch.utils.data import default_collate, DataLoader
import sys, os

sys.path.append(os.path.abspath('..'))

from callbacks.DDPM_callback import noisify


def collate_dict(dataset: Dataset) -> Callable:
    get = itemgetter(*dataset.features)

    def _f(batch: List) -> Any:
        return get(default_collate(batch))
    return _f


def collate_ddpm(b, alphabar=None, xl=None): return noisify(default_collate(b)[xl], alphabar)
def dl_ddpm(ds, bs=None): return DataLoader(ds, batch_size=bs, collate_fn=collate_ddpm, num_workers=4)