import numpy as np
from util import unitize
from util import norm
from util import bound
from util import vec_proj
from util import rodriguez_axis_angle
from util import array_from_yaml
from scipy import constants
from typing import Optional



class Guidance:


    def __init__(self) -> None:
        self.phase_id = 0


    @staticmethod
    def pronav(t, state: dict, args:dict, **kwargs):
        var_context = [state, kwargs.get("config_vars", {})]
        rm = np.asarray(state.get("rm"))
        vm = np.asarray(state.get("vm"))
        r_targ = array_from_yaml(args.get("TARGET"), var_context)
        v_targ = array_from_yaml(args.get("TARGET_DOT", [0, 0, 0]), var_context)
        N = float(args.get("N", 4.0))

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
        G_LIMIT = args.get("G_LIMIT", [])
        TIME_CONST = float(args.get("TIME_CONST")) #type:ignore
        VEL_DESIRED = args.get("DESIRED") #type:ignore
        ROT_AXIS = args.get("ROT_AXIS")

        # eval desired velocity vector from config file
        var_context = [state, kwargs.get("config_vars", {})]
        vm = np.asarray(state.get("vm"))
        vd = array_from_yaml(VEL_DESIRED, var_context)

        vm_hat = unitize(vm) 
        vd_hat = unitize(vd)

        if ROT_AXIS:
            rot_axis = np.asarray(ROT_AXIS)
            # project vd and vm hat
            vd_hat = unitize([vd_hat[0], vd_hat[1], 0])
            vm_hat = unitize([vm_hat[0], vm_hat[1], 0])
            # vd_hat = unitize(np.cross(vd_hat, rot_axis))
            # vm_hat = unitize(np.cross(vm_hat, rot_axis))
        else:
            rot_axis = np.cross(vd_hat, vm_hat) #type:ignore
            if norm(rot_axis) == 0.0:   # edge case when vd & vm are parallel
                if abs(vd_hat[2]) > 0:
                    rot_axis = np.array([1, 0, 0])
                elif abs(vd_hat[1]) > 0 or abs(vd_hat[0]) > 0:
                    rot_axis = np.array([0, 0, 1])

        vd_vm_dot = bound(np.dot(vd_hat, vm_hat), -1.0, 1.0) # protect against invalid arccos
        ang = bound(np.arccos(vd_vm_dot), -np.pi, np.pi) #type:ignore
        ac_hat = unitize(np.cross(vm_hat, unitize(rot_axis))) #type:ignore
        turn_accel = ang * (norm(vm) /  TIME_CONST) #type:ignore
        if len(G_LIMIT) == 2:
            turn_accel = bound(turn_accel, G_LIMIT[0] * constants.g, G_LIMIT[1] * constants.g)
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
        DESIRED_ALT = float(args.get("DESIRED")) #type:ignore
        ANGLE_LIMIT_DEG = float(args.get("ANGLE_LIMIT_DEG", 90.0))
        ALT_TIME_CONST = float(args.get("TIME_CONST")) #type:ignore
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
        ang_bound = np.radians(ANGLE_LIMIT_DEG)
        ang_err = (K_ANG / speed) * (DESIRED_ALT - alt) #type:ignore
        ang_err = bound(ang_err, -ang_bound, ang_bound)
        azimuth_proj = unitize(np.array([vm[0], vm[1], 0])) #type:ignore
        zz = np.array([0, 0, 1])
        rot_axis = unitize(np.cross(azimuth_proj, zz))
        R = rodriguez_axis_angle(rot_axis, ang_err)
        vd = R @ azimuth_proj

        # pass to PN
        PN_args = {}
        PN_args["TIME_CONST"] = args.get("KD")
        PN_args["DESIRED"] = vd
        ac = Guidance.PN(t, state, PN_args, **kwargs)
        return ac


    @staticmethod
    def speed_controller(t, state, args, **kwargs):
        desired = args.get("DESIRED")
        value = state.get("speed")
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
    def accelerate(t, state, args, **kwargs):
        vm = state.get("vm")
        burn_G = args.get("G")
        return unitize(vm) * (burn_G * constants.g)


    def run(self, t, rm, vm, r_targ, config):
        ac = np.zeros((3,))
        range = norm(rm)
        speed = norm(vm)
        east = rm[0]
        north = rm[1]
        alt = rm[2]
        east_dot = vm[0]
        north_dot = vm[1]
        alt_dot = vm[2]
        state = {
                "rm": rm,
                "vm": vm,
                "range": range,
                "speed": speed,
                "alt": alt,
                "alt_dot": alt_dot,
                "north": north,
                "east": east,
                "north_dot": north_dot,
                "east_dot": east_dot,
                }
        v_targ = np.zeros((3,))

        ignores = ["condition_next", "enable_drag", "enable_gravity"]

        if "guidance" in config:
            gd_vars = config["guidance"].get("vars", {})
            globals().update(gd_vars)

            gd_phase = config["guidance"]["phase"][self.phase_id]
            gd_condition_next = gd_phase.get("condition_next")
            for func_name in gd_phase:
                if func_name in ignores:
                    continue
                gd_func = self.__getattribute__(func_name)
                gd_args = gd_phase[func_name]

                ac += gd_func(t, state, gd_args, ac=ac, config_vars=gd_vars)

            if gd_condition_next and eval(gd_condition_next):
                # check if next phase is defined
                # else hold current phase
                next_phase = self.phase_id + 1
                if next_phase in config["guidance"]["phase"]:
                    self.phase_id += 1

        return ac
