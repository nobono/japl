


def flatten_list(_list) -> list:
    ret = []
    for item in _list:
        if not hasattr(item, "__len__"):
            ret += [item]
        else:
            ret += flatten_list(item)
    return ret
