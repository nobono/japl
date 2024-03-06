import numpy as np
from util import unitize
from util import inv
import guidance

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


    def popup_maneuver(self, rm, vm, r_targ):
        #######################
        # OLD
        #######################
        # vm = unitize(X[3:6])
        # base2 = [1, 0, 0]
        # Rt = create_Rt(
        #         vm,
        #         np.cross(vm, base2),
        #         base2
        #         )
        # ac = np.zeros((3,))
        # ac[1] = -1*np.sin(.3 * t)
        # return Rt @ ac
        #######################
        START_POP_RANGE = 6.5e3
        STOP_POP_ALT = 90
        START_DIVE_RANGE = 8e3
        STOP_DIVE_ALT = 30 # 60
        match self.gd_phase:
            case 0 :
                r_pop = np.array([0, START_POP_RANGE, 90])
                ac = guidance.pronav(rm, vm, r_pop, np.array([0, 0, 0]), N=4)
                if rm[2] >= STOP_POP_ALT:
                    self.gd_phase += 1
            case 1 :
                r_pop = np.array([0, START_DIVE_RANGE, 10])
                ac = guidance.pronav(rm, vm, r_pop, np.array([0, 0, 0]), N=3)
                if rm[2] <= STOP_DIVE_ALT:
                    self.gd_phase += 1
            case 2 :
                #######################
                # OLD
                #######################
                # Kp = 80.0
                # Kp_rz = 0.0006
                # rz_err = Kp_rz * (max(min(10 - rm[2], 10), -10) / 10)
                # vmd_hat = unitize([0, 1, rz_err])
                # vm_hat = unitize(vm)
                # vm_err = vmd_hat - vm_hat
                # ac = ac + Kp * vm_err
                #######################
                r_pop = np.array([0, 13e3, 10])
                ac = guidance.pronav(rm, vm, r_pop, np.array([0, 0, 0]), N=40)
                if rm[1] > 12e3:
                    self.gd_phase += 1
            case 3 :
                ac = guidance.pronav(rm, vm, r_targ, np.array([0, 0, 0]), N=4)
            case _ :
                ac = np.array([0, 0, 0])
                pass
        return ac
