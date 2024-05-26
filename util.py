import os
import yaml
import numpy as np
from typing import Any

CONFIGS_DIR = os.path.join(os.getcwd(), "configs")


# for eval
sin = np.sin
cos = np.cos
tan = np.tan
atan = np.arctan
atan2 = np.arctan2
acos = np.arccos
asin = np.arcsin
degrees = np.degrees
radians = np.radians
pi = np.pi



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
    # with open(os.path.join(CONFIGS_DIR, filename), "r") as f:
    with open(filename, "r") as f:
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
    # norm = np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
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


def rodriguez_axis_angle(axis: np.ndarray, ang: float) -> np.ndarray:
    """Uses Rodriguez Rotation formula and returns a 
    rotation matrix"""
    N = len(axis)
    S = skew(axis)
    R = np.eye(N) + np.sin(ang) * S + (1 - np.cos(ang)) * (S @ S)
    return R


def create_rot_from_vecs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Uses Rodriguez Rotation formula to create rotation
    matrix between two vectors"""
    assert len(a) > 1
    assert len(a) == len(b)
    N = len(a)
    axis = np.cross(a, b)
    ang = np.dot(a, b)
    S = skew(unitize(axis))
    R = np.eye(N) + norm(axis) * S + (1 - ang) * (S @ S)
    return R


def create_rot_mat(vm: np.ndarray):
    """creates rotation matrix from velocity vector
    (vehicle body rotation matrix)"""
    return create_rot_from_vecs(np.array([0, 1, 0]), vm)


def skew(vec: np.ndarray):
    """creates skew matrix from vector"""
    return np.array([
        [0, -vec[2], vec[1]],
        [vec[2], 0, -vec[0]],
        [-vec[1], vec[0], 0]
    ])


def vec_proj(a: np.ndarray, b: np.ndarray):
    """vector projection of vec a onto vec b"""
    return unitize(b) * (np.dot(a, b) / norm(b))


def array_from_yaml(value: Any, var_context: list[dict]):
    # extract values from state
    for context in var_context:
        for key, val in context.items():
            locals()[key] = val
    # eval list or each index of list
    if isinstance(value, str):
        vd = eval(value)
    else:
        vd = []
        for i in value: #type:ignore
            if isinstance(i, str):
                vd.append(eval(i))
            else:
                vd.append(i)
        vd = np.asarray(vd)
    return vd

