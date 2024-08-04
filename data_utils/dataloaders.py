from .collate_dict import collate_dict
from torch.utils.data import DataLoader


class DataLoaders:
    def __init__(self, *dataloaders):
        self.train, self.valid = dataloaders[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, num_workers, as_tuple=True):
        """
        from dataset dict
        """
        return cls(*[DataLoader(ds, batch_size, collate_fn=collate_dict(ds), drop_last=True, num_workers=2) 
                     for ds in dd.values()
                     ])