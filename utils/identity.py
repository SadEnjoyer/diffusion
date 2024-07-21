def identity(*args):
    if not args: return
    x, *args = args
    return (x,)+tuple(args) if args else x