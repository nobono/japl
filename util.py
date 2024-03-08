import os
import yaml
import numpy as np

CONFIGS_DIR = os.path.join(os.getcwd(), "configs")



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


def read_config_file(filename: str):
    with open(os.path.join(CONFIGS_DIR, filename), "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise Exception(e)
    return data


def norm(vec: np.ndarray) -> float:
    return np.linalg.norm(vec) #type:ignore


def bound(val, lower, upper):
    return min(max((val), lower), upper)


def unitize(vec):
    norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    return vec / norm


def inv(mat):
    return np.linalg.inv(mat)


def create_C_rot(vm):
    vm = unitize(vm)
    bvec2 = np.array([0, 0, 1])
    return inv(np.array([
        vm,
        np.cross(vm, bvec2),
        bvec2
        ]))


def create_rot_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Uses Rodriguez Rotation formula"""
    assert len(a) == len(b)
    N = len(a)
    rot_axis = np.cross(a, b)
    rot_ang = np.dot(a, b)
    S = skew(a)
    R = np.eye(N) + S + (S @ S) * ((1 - rot_ang) / (np.linalg.norm(rot_axis)**2))
    return R


def skew(vec: np.ndarray):
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])


def check_for_events(t_events):
    for event in t_events:
        if len(event):
            return True
    return False
