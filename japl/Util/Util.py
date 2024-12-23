import numpy as np
from typing import Any, Iterable
from typing import get_args
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


def iter_type_check(iter: Iterable, typehint) -> bool:
    if isinstance(iter, Iterable):  # type:ignore
        list_type = get_args(typehint)
        if len(list_type):
            return all(isinstance(i, list_type) for i in iter)
        else:
            return False
    else:
        return False


def unitize(vec):
    # norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm
