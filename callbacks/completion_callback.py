from .basic_callback import *

class CompletionCB(Callback):
    def before_fit(self): self.count = 0
    def after_batch(self): self.count+=1
    def after_fit(self): print(f'Completed {self.count} batches')