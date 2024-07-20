from operator import itemgetter
from typing import Any, List
from collections.abc import Callable
from datasets import Dataset

def collate_dict(dataset: Dataset) -> Callable:
    get = itemgetter(*dataset.features)
    def _f(batch: List) -> Any: 
        return get(default_collate(batch))
    return _f

class CollateDictWindows():
    def __init__(self):
        get = itemg