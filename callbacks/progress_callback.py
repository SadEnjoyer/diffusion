from .basic_callback import Callback
from .metrics_callback import MetricsCB
from fastprogress import progress_bar, master_bar
import fastcore.all as fc


class ProgressCB(Callback):
    order = MetricsCB.order + 1
    def __init__(self, plot=False): self.plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn): learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)

    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses:
                self.mbar.update_graph([
                    [fc.L.range(self.losses), self.losses],
                    [fc.L.range(learn.epoch).map(lambda x: (x + 1) * len(learn.dls.train)), self.val_losses]
                ])

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'):
                self.val_losses.append(learn.metrics.all_metrics['loss'].compute())
                self.mbar.update_graph([
                    [fc.L.range(self.losses), self.losses],
                    [fc.L.range(learn.epoch + 1).map(lambda x: (x + 1) * len(learn.dls.train)), self.val_losses]
                ])