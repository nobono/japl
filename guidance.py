import numpy as np
from util import unitize
from scipy.linalg import norm



def pronav(rm, vm, r_targ, v_targ=np.zeros((3,)), N=4.0):
    v_r = v_targ - vm
    r = r_targ - rm
    omega = np.cross(r, v_r) / np.dot(r, r)
    ac = N * np.cross(v_r, omega)
    return ac


def PN(vm, vd) -> np.ndarray:
    vm_hat = unitize(vm) 
    vd_hat = unitize(vd)
    ac = np.cross(np.arcsin(np.cross(vm_hat, vd_hat)) / norm(vm), vd_hat)
    return ac


def p_controller(desired, value, gain):
    return gain * (desired - value)

