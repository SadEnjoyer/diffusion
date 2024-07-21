from operator import itemgetter
from typing import Any, List
from collections.abc import Callable
from datasets import Dataset
from torch.utils.data import default_collate

def collate_dict(dataset: Dataset) -> Callable:
    get = itemgetter(*dataset.features)
    def _f(batch: List) -> Any: 
        return get(default_collate(batch))
    return _f
