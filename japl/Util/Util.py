import timeit



def profile(func):
    def wrapped(*args, **kwargs):
        start_time = timeit.default_timer()
        res = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print("-" * 50)
        print(f"{func.__name__} executed in {end_time - start_time} seconds")
        print("-" * 50)
        return res
    return wrapped


def flatten_list(_list: list|tuple) -> list:
    ret = []
    for item in _list:
        if not hasattr(item, "__len__") or isinstance(item, str):
            ret += [item]
        else:
            ret += flatten_list(item)
    return ret
