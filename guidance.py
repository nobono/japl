import numpy as np
from util import unitize
from util import norm
from util import bound
from util import vec_proj
from scipy import constants



class Guidance:


    def __init__(self) -> None:
        self.phase_id = 0


    @staticmethod
    def pronav(rm, vm, r_targ, v_targ=np.zeros((3,)), N=4.0):
        v_r = v_targ - vm
        r = r_targ - rm
        omega = np.cross(r, v_r) / np.dot(r, r)
        ac = N * np.cross(v_r, omega)
        return ac


    @staticmethod
    def PN(vd, vm, TC, bounds: list=[]) -> np.ndarray:
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
        turn_accel = ang * (norm(vm) /  TC)
        if len(bounds) == 2:
            turn_accel = bound(turn_accel, bounds[0], bounds[1])
        ac = ac_hat * turn_accel
        ###################################
        # ac = np.cross(np.arcsin(np.cross(vm_hat, vd_hat)) / norm(vm), vd_hat)
        # ac = ac_hat * norm(vm)
        ###################################
        return ac


    @staticmethod
    def p_controller(desired, value, gain):
        return gain * (desired - value)


    @staticmethod
    def pd_controller(desired, value, rate, KP, KD, bounds: list=[]):
        desired_rate = KP * (desired - value)
        if len(bounds) == 2:
            desired_rate = bound(desired_rate, bounds[0], bounds[1])
        out = KD * (desired_rate - rate)
        return out


    @staticmethod
    def alt_controller(t, state: dict, args: dict, **kwargs):
        ALT_RATE_LIMIT = float(args["ALT_RATE_LIMIT"])
        DESIRED_ALT = float(args["DESIRED_ALT"])
        ALT_TIME_CONST = float(args["ALT_TIME_CONST"])
        KP = 1.0 / ALT_TIME_CONST
        KD = float(args["KD"])

        alt = state["alt"]
        alt_dot = state["alt_dot"]
        # C_i_v = create_C_rot(state["vm"])

        bounds = [-ALT_RATE_LIMIT, ALT_RATE_LIMIT]
        ac_alt = Guidance.pd_controller(DESIRED_ALT, alt, alt_dot, KP, KD, bounds=bounds)
        # ac = C_i_v @ np.array([0, 0, ac_alt])
        ac = np.array([0, 0, ac_alt])
        return ac