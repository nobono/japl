import numpy as np



class ID:
    def __init__(self, state_array: list[str]) -> None:
        assert isinstance(state_array, list)
        for i, state in enumerate(state_array):
            self.__setattr__(state, i)


class State:
    def __init__(self, sol, t) -> None:
        self.t = t
        self.xpos = sol[:, 0]
        self.ypos = sol[:, 1]
        self.xvel = sol[:, 2]
        self.yvel = sol[:, 3]
        self.xacc = sol[:, 4]
        self.yacc = sol[:, 5]
        self.xjerk = sol[:, 6]
        self.yjerk = sol[:, 7]
