import numpy as np
from util import unitize
from util import norm
from util import bound



def pronav(rm, vm, r_targ, v_targ=np.zeros((3,)), N=4.0):
    v_r = v_targ - vm
    r = r_targ - rm
    omega = np.cross(r, v_r) / np.dot(r, r)
    ac = N * np.cross(v_r, omega)
    return ac


def PN(vd, vm) -> np.ndarray:
    vm_hat = unitize(vm) 
    vd_hat = unitize(vd)
    ac = np.cross(np.arcsin(np.cross(vm_hat, vd_hat)) / norm(vm), vd_hat)
    # ac_hat = unitize(vd - vm)
    # np.arcsin(norm(np.cross(vm_hat, vd_hat)))
    return ac


def p_controller(desired, value, gain):
    return gain * (desired - value)


def pd_controller(desired, value, rate, KP, KD, bounds: list=[]):
    desired_rate = KP * (desired - value)
    if len(bounds) == 2:
        desired_rate = bound(desired_rate, bounds[0], bounds[1])
    out = KD * (desired_rate - rate)
    return out

