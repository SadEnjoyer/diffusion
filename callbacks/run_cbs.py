from operator import attrgetter

def run_cbs(cbs, method_nm):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method()