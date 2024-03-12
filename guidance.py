import numpy as np
from util import unitize
from util import norm
from util import bound
from util import vec_proj
from util import rodriguez_axis_angle
from scipy import constants


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


class Guidance:


    def __init__(self) -> None:
        self.phase_id = 0


    @staticmethod
    def pronav(t, state: dict, args:dict, **kwargs):
        # rm, vm, r_targ, v_targ=np.zeros((3,)), N=4.0
        rm = state["rm"]
        vm = state["vm"]
        r_targ = kwargs["r_targ"]
        v_targ = kwargs["v_targ"]
        N = args["N"]

        v_r = v_targ - vm
        r = r_targ - rm
        omega = np.cross(r, v_r) / np.dot(r, r)
        ac = N * np.cross(v_r, omega)
        return ac


    @staticmethod
    # def PN(t, vd, vm, TC, bounds: list=[]) -> np.ndarray:
    def PN(t, state: dict, args: dict, **kwargs) -> np.ndarray:
        """
        @args
        vd - desired velocity vector
        vm - velocity vector
        G_LIMIT - 
        """
        TIME_CONST = args.get("TIME_CONST", None)
        if TIME_CONST is None:
            raise Exception("guidance.PN() required TIME_CONST argument")

        vm = state.get("vm")
        if "VEL_HAT_DESIRED" in args:
            _vd = args.get("VEL_HAT_DESIRED")
            if isinstance(_vd, str):
                vd = eval(_vd)
            else:
                vd = []
                for i in _vd:
                    if isinstance(i, str):
                        vd.append(eval(i))
                    else:
                        vd.append(i)
                vd = np.asarray(vd)
        if vd is None:
            raise Exception("guidance.PN() required VEL_HAT_DESIRED argument")

        bounds = args.get("bounds", [])
        vm_hat = unitize(vm) 
        vd_hat = unitize(vd)
        rot_axis = np.cross(vd_hat, vm_hat)
        if norm(rot_axis) == 0.0:   # edge case when vd & vm are parallel
            if abs(vd_hat[2]) > 0:
                rot_axis = np.array([1, 0, 0])
            elif abs(vd_hat[1]) > 0 or abs(vd_hat[0]) > 0:
                rot_axis = np.array([0, 0, 1])

        ang = np.arccos(bound(np.dot(vd_hat, vm_hat), -1.0, 1.0))
        ac_hat = unitize(np.cross(vm_hat, unitize(rot_axis)))
        turn_accel = ang * (norm(vm) /  TIME_CONST)
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
        DESIRED_ALT = float(args["DESIRED_ALT"])
        ALT_TIME_CONST = float(args["TIME_CONST"])
        K_ANG = 1.0 / ALT_TIME_CONST # (rad / s)

        alt = state.get("alt")
        alt_dot = state.get("alt_dot")
        vm = state.get("vm")
        speed = state.get("speed")

        # old method
        ###################
        # KD = float(args["KD"])
        # bounds = [-ALT_RATE_LIMIT, ALT_RATE_LIMIT]
        # ac_alt = Guidance.pd_controller(DESIRED_ALT, alt, alt_dot, KP, KD, bounds=bounds)
        # ac = C_i_v @ np.array([0, 0, ac_alt])
        # ac = np.array([0, 0, ac_alt])
        ###################
        ang_err = (K_ANG / speed) * (DESIRED_ALT - alt)
        ang_err = bound(ang_err, -radians(90), radians(90))
        azimuth_proj = unitize(np.array([vm[0], vm[1], 0]))
        zz = np.array([0, 0, 1])
        rot_axis = unitize(np.cross(azimuth_proj, zz))
        R = rodriguez_axis_angle(rot_axis, ang_err)
        vd = R @ azimuth_proj

        # pass to PN
        PN_args = {}
        PN_args["TIME_CONST"] = args.get("KD")
        PN_args["VEL_HAT_DESIRED"] = vd
        ac = Guidance.PN(t, state, PN_args, **kwargs)
        return ac


    @staticmethod
    def speed_controller(t, state, args, **kwargs):
        desired = args.get("DESIRED")
        value = state.get(args.get("VALUE"))
        ac = kwargs.get("ac")
        vm = state.get("vm")
        if "GAIN" in args:
            gain = args.get("GAIN")
        elif "TIME_CONST" in args:
            gain = 1.0 / args.get("TIME_CONST")
        else:
            raise Exception("GAIN or TIME_CONST must be an argument to guidance.p_controller()")
        accel_mag = gain * (desired - value)
        return unitize(vm) * accel_mag


    @staticmethod
    def burn_at_g(t, state, args, **kwargs):
        vm = state.get("vm")
        burn_G = args.get("G")
        return unitize(vm) * burn_G
