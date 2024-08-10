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

acc = (force) / mass
angacc = inertia.inv() * (torque)

# gravity
gacc_new = -atmosphere.grav_accel(pos_z) #type:ignore
gravity = Matrix([0, 0, gacc_new])

##################################################
# AeroTable
##################################################
sos = atmosphere.speed_of_sound(pos_z) #type:ignore
mach_new = speed / sos #type:ignore

# calc angle of attack: (pitch_angle - flight_path_angle)
vel_hat = vel / speed   # flight path vector

# projection vel_hat --> x-axis
zx_plane_norm = Matrix([0, 1, 0])
vel_hat_zx = ((vel_hat.T @ zx_plane_norm) / zx_plane_norm.norm())[0] * zx_plane_norm
vel_hat_proj = vel_hat - vel_hat_zx

# get Trait-bryan angles (yaw, pitch, roll)
yaw_angle, pitch_angle, roll_angle = RotationSymbolic.quat_to_tait_bryan_sym(quat)

# angle between proj vel_hat & xaxis
x_axis_inertial = Matrix([1, 0, 0])
flight_path_angle = sign(vel_hat_proj[2]) * VecSymbolic.vec_ang_sym(vel_hat_proj, x_axis_inertial) #type:ignore
alpha_new = pitch_angle - flight_path_angle # angle of attack
phi_new = roll_angle

iota = rad(0.1)
CLMB = -aerotable.get_CLMB_Total(alpha, phi, mach, iota) #type:ignore
CNB = aerotable.get_CNB_Total(alpha, phi, mach, iota) #type:ignore
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
force_aero = Matrix([0, 0, 0])
torque_aero = Matrix([0, 0, 0])

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
        # (alpha,             alpha_new),
        # (phi,               phi_new),
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
    DirectUpdate(gacc, gacc_new), #type:ignore
    DirectUpdate(speed, vel.norm()),
    DirectUpdate(mach, mach_new), #type:ignore
    DirectUpdate(alpha, alpha_new),
    DirectUpdate(phi, phi_new),
    ])

input = Matrix([
    DirectUpdate(force, force_aero),
    # force,
    DirectUpdate(torque, torque_aero),
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

