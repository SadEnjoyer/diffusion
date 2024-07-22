from functools import partial


class Hook():
    def __init__(self, model, function): self.hook = model.register_forward_hook(partial(function, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()


class Hooks(list):
    def __init__(self, methods, func): super().__init__([Hook(m, func) for m in methods])
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()

    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)

    def remove(self):
        for h in self: h.remove()
