import numpy as np
from util import unitize
from util import norm
from util import bound
from util import vec_proj



def pronav(rm, vm, r_targ, v_targ=np.zeros((3,)), N=4.0):
    v_r = v_targ - vm
    r = r_targ - rm
    omega = np.cross(r, v_r) / np.dot(r, r)
    ac = N * np.cross(v_r, omega)
    return ac


def PN(vd, vm, G_LIMIT, bounds: list=[]) -> np.ndarray:
    """
    @args
    vd - desired velocity vector
    vm - velocity vector
    G_LIMIT - 
    """
    vm_hat = unitize(vm) 
    vd_hat = unitize(vd)
    rot_axis = np.cross(vd_hat, vm_hat)
    if norm(rot_axis) == 0.0:   # edge case when vd & vm are parallel
        if abs(vd_hat[2]) > 0:
            rot_axis = np.array([1, 0, 0])
        elif abs(vd_hat[1]) > 0 or abs(vd_hat[0]) > 0:
            rot_axis = np.array([0, 0, 1])

    ang = np.arccos(np.dot(vd_hat, vm_hat))
    ac_hat = unitize(np.cross(vm_hat, unitize(rot_axis)))
    turn_accel = ang * (norm(vm) / G_LIMIT)
    if len(bounds) == 2:
        turn_accel = bound(turn_accel, bounds[0], bounds[1])
    ac = ac_hat * turn_accel
    ###################################
    # ac = np.cross(np.arcsin(np.cross(vm_hat, vd_hat)) / norm(vm), vd_hat)
    # ac = ac_hat * norm(vm)
    ###################################
    return ac


def p_controller(desired, value, gain):
    return gain * (desired - value)


def pd_controller(desired, value, rate, KP, KD, bounds: list=[]):
    desired_rate = KP * (desired - value)
    if len(bounds) == 2:
        desired_rate = bound(desired_rate, bounds[0], bounds[1])
    out = KD * (desired_rate - rate)
    return out

