from .collate_dict import collate_dict
from torch.utils.data import DataLoader


class DataLoaders:
    def __init__(self, *dataloaders):
        self.train, self.valid = dataloaders[:2]

    @classmethod
    def from_dd(cls, dataset_dict, batch_size, num_workers, as_tuple=True):
        """
        from dataset dict
        """
        return cls(*[DataLoader(ds, batch_size, collate_fn=collate_dict(ds), drop_last=True, num_workers=num_workers)
                     for ds in dataset_dict.values()
                     ])


class MultDL:
    def __init__(self, dl, mult=2): self.dl, self.mult = dl, mult
    def __len__(self): return len(self.dl) * self.mult

    def __iter__(self):
        for o in self.dl:
            for i in range(self.mult): yield o