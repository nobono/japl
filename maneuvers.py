import numpy as np

# ---------------------------------------------------

from util import unitize
from util import inv
from util import create_C_rot
from util import bound

# ---------------------------------------------------

import guidance

# ---------------------------------------------------

from scipy import constants
from scipy.linalg import norm

# ---------------------------------------------------



class Maneuvers:
    

    def __init__(self) -> None:
        self.gd_phase = 0


    @staticmethod
    def weave_maneuver(t, X):
        vm = unitize(X[3:6])
        base2 = [0, 0, 1]
        mat = np.array([vm, np.cross(vm, base2), base2])
        Rt = inv(mat)
        ac = np.array([0, np.sin(0.5 * t), 0])
        return Rt @ ac


    def uo_dive(self, rm, vm, r_targ):
        # ac = np.zeros((3,))
        CRUISE_ALT = 500.0
        ALT_RATE_LIMIT = 10.0
        r_alt = rm[2]

        match self.gd_phase:
            case 0 :
                # Entry
                K_P = 0.2
                K_D = 0.8
                alt_dot = vm[2]
                ascend_rate = K_P * (CRUISE_ALT - r_alt)
                ac_alt = K_D * (ascend_rate - alt_dot)
                ac_alt = bound(ac_alt, -ALT_RATE_LIMIT, ALT_RATE_LIMIT)
                C_i_v = create_C_rot(vm)
                ac = C_i_v @ np.array([0, 0, ac_alt])
                # if r_range <= START_ASCEND_RANGE:
                #     self.gd_phase += 1
            case _ :
                ac = np.zeros((3,))
                raise Exception("unhandled event")

        GLIMIT = 14.0
        if norm(ac) > GLIMIT:
            ac = unitize(ac) * GLIMIT
        return ac


    def popup(self, rm, vm, r_targ):
        ASCEND_SPEED = 400.0
        CRUISE_ALT = 10.0
        START_ASCEND_RANGE = 45e3
        ASCEND_RATE_LIMIT = 200.0
        START_DIVE_ALT = 200.0
        #####################
        START_ASCEND_RANGE_2 = 32e3
        ASCEND_RATE_LIMIT_2 = 200.0
        START_DIVE_ALT_2 = 120.0
        #####################
        START_TERMINAL_RANGE = 8e3
        #####################
        K_P = 0.05
        K_D = 0.06
        r_range = norm(rm)
        r_alt = rm[2]
        match self.gd_phase:
            case 0 :
                # Entry
                K_P *= 2.0
                K_D *= 3
                alt_dot = vm[2]
                ascend_rate = max(K_P * (CRUISE_ALT - r_alt), -ASCEND_RATE_LIMIT)
                ac_alt = K_D * (ascend_rate - alt_dot)
                C_i_v = create_C_rot(vm)
                ac = C_i_v @ np.array([0, 0, ac_alt])
                if r_range <= START_ASCEND_RANGE:
                    self.gd_phase += 1
            case 1 :
                # Ascend
                K_P *= 2.0
                K_D *= 3
                alt_dot = vm[2]
                ascend_rate = max(K_P * (START_DIVE_ALT - r_alt), -ASCEND_RATE_LIMIT)
                ac_alt = K_D * (ascend_rate - alt_dot)
                C_i_v = create_C_rot(vm)
                ac = C_i_v @ np.array([0, 0, ac_alt])
                if r_alt >= START_DIVE_ALT:
                    self.gd_phase += 1
            case 2 :
                # Descend
                K_P *= 3.5
                K_D *= 10.0
                ac = np.zeros((3,))
                alt_dot = vm[2]
                ascend_rate = min(K_P * (CRUISE_ALT - r_alt), ASCEND_RATE_LIMIT)
                ac_alt = K_D * (ascend_rate - alt_dot)
                C_i_v = create_C_rot(vm)
                ac = C_i_v @ np.array([0, 0, ac_alt])
                if r_range <= START_ASCEND_RANGE_2:
                    self.gd_phase += 1
            case 3 :
                # Ascend
                K_P *= 3.0
                K_D *= 3
                alt_dot = vm[2]
                ascend_rate = max(K_P * (START_DIVE_ALT_2 - r_alt), -ASCEND_RATE_LIMIT_2)
                ac_alt = K_D * (ascend_rate - alt_dot)
                C_i_v = create_C_rot(vm)
                ac = C_i_v @ np.array([0, 0, ac_alt])
                if r_alt >= START_DIVE_ALT_2:
                    self.gd_phase += 1
            case 4 :
                # Descend
                K_P *= 3.5
                K_D *= 10.0
                ac = np.zeros((3,))
                alt_dot = vm[2]
                ascend_rate = min(K_P * (CRUISE_ALT - r_alt), ASCEND_RATE_LIMIT_2)
                ac_alt = K_D * (ascend_rate - alt_dot)
                C_i_v = create_C_rot(vm)
                ac = C_i_v @ np.array([0, 0, ac_alt])
                if r_range <= START_TERMINAL_RANGE:
                    self.gd_phase += 1
            case 5 :
                ac = guidance.pronav(rm, vm, r_targ, np.zeros((3,)), N=4)
            case _ :
                ac = np.zeros((3,))
                raise Exception("unhandled event")

        GLIMIT = 14.0
        if norm(ac) > (GLIMIT * constants.g):
            ac = unitize(ac) * (GLIMIT * constants.g)
        return ac
