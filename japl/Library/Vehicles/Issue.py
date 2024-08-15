import sympy as sp
from sympy import Matrix, Symbol, symbols
from sympy import sqrt, sign, rad
from japl import Model
from japl.Math.MathSymbolic import zero_protect_sym
from sympy import Function

from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.Aero.AeroTableSymbolic import AeroTableSymbolic
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl.Math import RotationSymbolic
from japl.Math import VecSymbolic



class MissileGeneric(Model):
    pass


################################################
# Debug
################################################
def print_sym(var, msg: str = "DEBUG "):
    print(msg, var)
    return var

debug_module = {
        "print_sym": print_sym,
        }
print_sym = Function("print_sym") #type:ignore

################################################
# Tables
################################################

atmosphere = AtmosphereSymbolic()
aero_file = "/home/david/work_projects/control/aeromodel/aeromodel_psb.mat"
# aero_file = "../../../aeromodel/aeromodel_psb.mat"
aerotable = AeroTableSymbolic(aero_file)

################################################
# Variables
################################################

t = symbols("t")
dt = symbols("dt")

pos_x = Function("pos_x", real=True)(t) #type:ignore
pos_y = Function("pos_y", real=True)(t) #type:ignore
pos_z = Function("pos_z", real=True)(t) #type:ignore

vel_x = Function("vel_x", real=True)(t) #type:ignore
vel_y = Function("vel_y", real=True)(t) #type:ignore
vel_z = Function("vel_z", real=True)(t) #type:ignore

angvel_x = Function("angvel_x", real=True)(t) #type:ignore
angvel_y = Function("angvel_y", real=True)(t) #type:ignore
angvel_z = Function("angvel_z", real=True)(t) #type:ignore

angacc_x = Symbol("angacc_x", real=True) #type:ignore
angacc_y = Symbol("angacc_y", real=True) #type:ignore
angacc_z = Symbol("angacc_z", real=True) #type:ignore

q_0 = Function("q_0", real=True)(t) #type:ignore
q_1 = Function("q_1", real=True)(t) #type:ignore
q_2 = Function("q_2", real=True)(t) #type:ignore
q_3 = Function("q_3", real=True)(t) #type:ignore

acc_x = Symbol("acc_x", real=True) #type:ignore
acc_y = Symbol("acc_y", real=True) #type:ignore
acc_z = Symbol("acc_z", real=True) #type:ignore

torque_x = Function("torque_x", real=True)(t) #type:ignore
torque_y = Function("torque_y", real=True)(t) #type:ignore
torque_z = Function("torque_z", real=True)(t) #type:ignore

force_x = Function("force_x", real=True)(t) #type:ignore
force_y = Function("force_y", real=True)(t) #type:ignore
force_z = Function("force_z", real=True)(t) #type:ignore

Ixx = Symbol("Ixx", real=True)
Iyy = Symbol("Iyy", real=True)
Izz = Symbol("Izz", real=True)

inertia = Matrix([
    [Ixx, 0, 0],
    [0, Iyy, 0],
    [0, 0, Izz],
    ])

gacc = Function("gacc", real=True)(t) #type:ignore

##################################################
# States
##################################################

pos = Matrix([pos_x, pos_y, pos_z])
vel = Matrix([vel_x, vel_y, vel_z])
angvel = Matrix([angvel_x, angvel_y, angvel_z])
quat = Matrix([q_0, q_1, q_2, q_3])
mass = symbols("mass")
speed = Symbol("speed", real=True) #type:ignore
mach = Function("mach", real=True)(t) #type:ignore

alpha = Symbol("alpha", real=True) #type:ignore
phi = Symbol("phi", real=True) #type:ignore
cg = Symbol("cg", real=True)

##################################################
# Inputs
##################################################

torque = Matrix([torque_x, torque_y, torque_z])
force = Matrix([force_x, force_y, force_z])

##################################################
# Equations Atmosphere
##################################################

acc = force / mass
angacc = inertia.inv() * torque

# gravity
gacc_new = -atmosphere.grav_accel(pos_z) #type:ignore
gravity = Matrix([0, 0, gacc_new])

##################################################
# AeroTable
##################################################
def aerotable_update_func(pos, vel, quat, cg, Iyy, speed, mach):
    sos = atmosphere.speed_of_sound(pos[2]) #type:ignore
    mach_new = speed / sos #type:ignore

    # calc angle of attack: (pitch_angle - flight_path_angle)
    vel_hat = vel / speed   # flight path vector

    # projection vel_hat --> x-axis
    zx_plane_norm = Matrix([0, 1, 0])
    vel_hat_zx = ((vel_hat.dot(zx_plane_norm)) / zx_plane_norm.norm()) * zx_plane_norm
    vel_hat_proj = vel_hat - vel_hat_zx

    # get Trait-bryan angles (yaw, pitch, roll)
    yaw_angle, pitch_angle, roll_angle = RotationSymbolic.quat_to_tait_bryan_sym(quat)

    # angle between proj vel_hat & xaxis
    x_axis_inertial = Matrix([1, 0, 0])
    flight_path_angle = sign(vel_hat_proj[2]) * VecSymbolic.vec_ang_sym(vel_hat_proj, x_axis_inertial) #type:ignore
    alpha_new = pitch_angle - flight_path_angle # angle of attack
    phi_new = roll_angle

    iota = sp.Float(0)
    CLMB = -aerotable.get_CLMB_Total(alpha_new, phi_new, mach, iota) #type:ignore
    CNB = aerotable.get_CNB_Total(alpha_new, phi_new, mach, iota) #type:ignore
    My_coef = CLMB + (cg - aerotable.get_MRC()) * CNB #type:ignore

    q = atmosphere.dynamic_pressure(vel, pos_z) #type:ignore
    Sref = aerotable.get_Sref()
    Lref = aerotable.get_Lref()
    My = My_coef * q * Sref * Lref

    force_z_aero = CNB * q * Sref #type:ignore
    force_aero = Matrix([0, 0, force_z_aero])

    torque_y_aero = My / Iyy
    torque_aero = Matrix([0, torque_y_aero, 0])

    # temp
    # force_aero = Matrix([0, 0, 1])
    # torque_aero = Matrix([0, 1, 0])

    extra = torque_y_aero

    return (alpha_new, phi_new, force_aero, torque_aero, extra)

######################
import numpy as np
from japl.Math import Rotation
from japl.Math import Vec
from japl.Aero.AeroTable import AeroTable
from japl.Aero.Atmosphere import Atmosphere
aero = AeroTable(aero_file)
atmos = Atmosphere()

def aerotable_update(pos, vel, quat, cg, Iyy, speed, mach):

    alt = pos[2]

    # calc angle of attack: (pitch_angle - flight_path_angle)
    vel_hat = vel / speed

    # projection vel_hat --> x-axis
    zx_plane_norm = np.array([0, 1, 0])
    vel_hat_zx = ((vel_hat @ zx_plane_norm) / np.linalg.norm(zx_plane_norm)) * zx_plane_norm
    vel_hat_proj = vel_hat - vel_hat_zx

    # get Trait-bryan angles (yaw, pitch, roll)
    yaw_angle, pitch_angle, roll_angle = Rotation.quat_to_tait_bryan(np.asarray(quat))

    # angle between proj vel_hat & xaxis
    x_axis_inertial = np.array([1, 0, 0])
    flight_path_angle = np.sign(vel_hat_proj[2]) * Vec.vec_ang(vel_hat_proj, x_axis_inertial)
    alpha_new = pitch_angle - flight_path_angle                     # angle of attack
    phi_new = roll_angle

    iota = 0.0
    CLMB = -aero.get_CLMB_Total(alpha_new, phi_new, mach, iota) #type:ignore
    CNB = aero.get_CNB_Total(alpha_new, phi_new, mach, iota) #type:ignore
    My_coef = CLMB + (cg - aero.get_MRC()) * CNB #type:ignore

    q = atmos.dynamic_pressure(vel, alt) #type:ignore
    Sref = aero.get_Sref()
    Lref = aero.get_Lref()
    My = My_coef * q * Sref * Lref

    force_z = CNB * q * Sref #type:ignore
    force_aero = np.array([0, 0, force_z])

    torque_y_new = My / Iyy
    torque_aero = np.array([0, torque_y_new, 0])

    # temp
    # force_aero = np.array([0, 0, 1])
    # torque_aero = np.array([0, 1, 0])

    extra = torque_y_new

    return (alpha_new, phi_new, force_aero, torque_aero,
            extra,
            )

# pos_ = np.array([0, 0, 0])
# vel_ = np.array([1500, 0, 0])
# quat_ = np.array([1, 0, 0, 0])
# cg_ = 1.4
# Iyy_ = 58.0
# speed_ = 1500
# mach_ = speed_ / 343.0

pos_ = np.array([ 3.00000000e+01,  0.00000000e+00,  9.99999902e+03])
vel_ = np.array([1.50000000e+03, 0.00000000e+00, -1.95366993e-01, ])
# angvel_ = [ 0.00000000e+00,  3.43229792e-04, 0.00000000e+00,]
quat_ = np.array([ 1.00000000e+00, -0.00000000e+00,  8.58074481e-07, 0.00000000e+00, ])
# mass_ = 1.33000000e+02
cg_ = 1.42000000e+00
# Ixx_ = 1.30900000e+00
Iyy_ = 5.82700000e+01
# Izz_ = 5.82700000e+01
# gacc_ = -9.81000000e+00
speed_ = 1.50000001e+03
mach_ = 5.00781791e+00
alpha_ = 1.31960810e-04
phi_ = 0.00000000e+00
# extra _ = 0.00000000e+00
iota_ = 0.0

iota = sp.Float(0)

(
alpha_new_,
phi_new_,
force_aero_,
torque_aero_,
extra_
) = aerotable_update(pos_, vel_, quat_, cg_, Iyy_, speed_, mach_)
# print(alpha_new_, phi_new_, force_aero_, torque_aero_, extra_)

(
alpha_new2_,
phi_new2_,
force_aero2_,
torque_aero2_,
extra2_
) = aerotable_update_func(pos, vel, quat, cg, Iyy, speed, mach)


from japl.Util.Desym import Desym
vars = (pos, vel, quat, cg, Iyy, speed, mach)
vars_ = (pos_, vel_, quat_, cg_, Iyy_, speed_, mach_)

modules = {}
modules.update(aerotable.modules)
modules.update(atmosphere.modules)
func = Desym(vars, extra2_, modules=modules)
ret = func(*vars_)

print("%.18f" % extra_)
print("%.18f" % ret)
print()
print(extra_ - ret)

quit()

aerotable_update_sym = Function("aerotable_update") #type:ignore
aero_module = {
        "aerotable_update_sym": aerotable_update,
        }
######################



##################################################
# Update Equations
##################################################

wx, wy, wz = angvel
Sw = Matrix([
    [ 0,   wx,  wy,  wz], #type:ignore
    [-wx,  0,  -wz,  wy], #type:ignore
    [-wy,  wz,   0, -wx], #type:ignore
    [-wz, -wy,  wx,   0], #type:ignore
    ])

pos_new = pos + vel * dt
vel_new = vel + (acc + gravity) * dt
angvel_new = angvel + angacc * dt
quat_new = quat + (-0.5 * Sw * quat) * dt
mass_new = mass

##################################################
# Differential Definitions
##################################################

pos_dot = pos_new.diff(dt)
vel_dot = vel_new.diff(dt)
angvel_dot = angvel_new.diff(dt)
quat_dot = quat_new.diff(dt)
mass_dot = mass_new.diff(dt)

defs = (
        (pos.diff(t),       pos_dot),
        (vel.diff(t),       vel_dot),
        (angvel.diff(t),    angvel_dot),
        (quat.diff(t),      quat_dot),
        (mass.diff(t),      mass_dot),
        )

##################################################
# Define State & Input Arrays
##################################################
# ------------------------------------------------
# NOTE: when adding to state array:
# ------------------------------------------------
# - Functions wrt. time will be diff'd and be
#   considered as part of the dynamics integration.
#
# - Symbols are treated as constants and get their
#   value from Simobj.init_state()
#
# - Any relationship can be defined in the definition
#   tuple above.
# ------------------------------------------------

# NOTE: speed needs to be directUpdate otherwise loss
# of precision

state = Matrix([
    pos,
    vel,
    angvel,
    quat,
    mass,
    cg,
    Ixx,
    Iyy,
    Izz,
    DirectUpdate(gacc, gacc), #type:ignore
    DirectUpdate(speed, vel.norm()),
    DirectUpdate(mach, mach_new), #type:ignore
    DirectUpdate(alpha, alpha_new),
    DirectUpdate(phi, phi_new),
    DirectUpdate("extra", extra),
    ])

input = Matrix([
    DirectUpdate(force, force_aero),
    DirectUpdate(torque, torque_aero),
    # force,
    # torque,
    ])

##################################################
# Define dynamics
##################################################

dynamics = state.diff(t)

##################################################
# Build Model
##################################################

model = MissileGeneric.from_expression(dt, state, input, dynamics,
                                         modules=[atmosphere.modules, aerotable.modules,
                                                  debug_module],
                                         definitions=defs)

