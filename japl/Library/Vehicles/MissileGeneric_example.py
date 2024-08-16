from sympy import Matrix, Symbol, symbols
from japl import Model
from sympy import Function
from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.BuildTools.DirectUpdate import DirectUpdate
# from japl.Math import RotationSymbolic
# from japl.Math import VecSymbolic



class MissileGeneric(Model):
    pass


t = symbols("t")
dt = symbols("dt")

pos_x = Function("pos_x", real=True)(t)  # type:ignore
pos_y = Function("pos_y", real=True)(t)  # type:ignore
pos_z = Function("pos_z", real=True)(t)  # type:ignore

vel_x = Function("vel_x", real=True)(t)  # type:ignore
vel_y = Function("vel_y", real=True)(t)  # type:ignore
vel_z = Function("vel_z", real=True)(t)  # type:ignore

angvel_x = Function("angvel_x", real=True)(t)  # type:ignore
angvel_y = Function("angvel_y", real=True)(t)  # type:ignore
angvel_z = Function("angvel_z", real=True)(t)  # type:ignore

q_0 = Function("q_0", real=True)(t)  # type:ignore
q_1 = Function("q_1", real=True)(t)  # type:ignore
q_2 = Function("q_2", real=True)(t)  # type:ignore
q_3 = Function("q_3", real=True)(t)  # type:ignore

acc_x = Function("acc_x", real=True)(t)  # type:ignore
acc_y = Function("acc_y", real=True)(t)  # type:ignore
acc_z = Function("acc_z", real=True)(t)  # type:ignore

torque_x = Function("torque_x", real=True)(t)  # type:ignore
torque_y = Function("torque_y", real=True)(t)  # type:ignore
torque_z = Function("torque_z", real=True)(t)  # type:ignore

Ixx = Symbol("Ixx", real=True)
Iyy = Symbol("Iyy", real=True)
Izz = Symbol("Izz", real=True)
inertia = Matrix([
    [Ixx, 0, 0],
    [0, Iyy, 0],
    [0, 0, Izz],
    ])

##################################################
# States
##################################################

pos = Matrix([pos_x, pos_y, pos_z])
vel = Matrix([vel_x, vel_y, vel_z])
angvel = Matrix([angvel_x, angvel_y, angvel_z])
quat = Matrix([q_0, q_1, q_2, q_3])
mass = symbols("mass")
gacc: Function = Function("gacc", real=True)(t)  # type:ignore
speed = Function("speed", real=True)(t)  # type:ignore
mach = Function("mach", real=True)(t)  # type:ignore

alpha = Function("alpha", real=True)(t)  # type:ignore
phi = Function("phi", real=True)(t)  # type:ignore
cg = Symbol("cg", real=True)

##################################################
# Inputs
##################################################

acc = Matrix([acc_x, acc_y, acc_z])
torque = Matrix([torque_x, torque_y, torque_z])

##################################################
# Equations for Aerotable / Atmosphere
##################################################

# gravity
atmosphere = AtmosphereSymbolic()
gacc_new = -atmosphere.grav_accel(pos_z)  # type:ignore

# speed
# speed_new = vel.norm()
# speed_dot = speed_new.diff(t)

# # mach
# sos = atmosphere.speed_of_sound(pos_z)  # type:ignore
# mach_new = speed / sos  # type:ignore

# # calc angle of attack: (pitch_angle - flight_path_angle)
# vel_hat = vel / speed_new   # flight path vector

# # projection vel_hat --> x-axis
# zx_plane_norm = Matrix([0, 1, 0])
# vel_hat_zx = ((vel_hat.T @ zx_plane_norm) / zx_plane_norm.norm())[0] * zx_plane_norm
# vel_hat_proj = vel_hat - vel_hat_zx

# # get Trait-bryan angles (yaw, pitch, roll)
# yaw_angle, pitch_angle, roll_angle = RotationSymbolic.quat_to_tait_bryan_sym(quat)

# # angle between proj vel_hat & xaxis
# x_axis_inertial = Matrix([1, 0, 0])
# flight_path_angle = sign(vel_hat_proj[2]) * VecSymbolic.vec_ang_sym(vel_hat_proj, x_axis_inertial)  # type:ignore
# alpha_new = pitch_angle - flight_path_angle # angle of attack
# phi_new = roll_angle

# # aerotable
# aero_file = "./aeromodel/aeromodel_psb.mat"
# # aero_file = "../../../aeromodel/aeromodel_psb.mat"
# aerotable = AeroTableSymbolic(aero_file)

# iota = rad(0.1)
# CLMB = -aerotable.get_CLMB_Total(alpha, phi, mach, iota)  # type:ignore
# CNB = aerotable.get_CNB_Total(alpha, phi, mach, iota)  # type:ignore
# My_coef = CLMB + (cg - aerotable.get_MRC()) * CNB  # type:ignore

# q = atmosphere.dynamic_pressure(vel, pos_z)  # type:ignore
# Sref = aerotable.get_Sref()
# Lref = aerotable.get_Lref()
# My = My_coef * q * Sref * Lref
# zforce = CNB * q * Sref  # type:ignore

# torque_y_new = My / Iyy
# torque_new = Matrix([torque_x, torque_y_new, torque_z])

# acc_z_new = zforce / mass
# acc_new = Matrix([acc_x, acc_y, acc_z_new])

##################################################
# Update Equations
##################################################

wx, wy, wz = angvel
Sw = Matrix([
    [0, wx, wy, wz],  # type:ignore
    [-wx, 0, -wz, wy],  # type:ignore
    [-wy, wz, 0, -wx],  # type:ignore
    [-wz, -wy, wx, 0],  # type:ignore
    ])

pos_new = pos + vel * dt
vel_new = vel + (acc + Matrix([0, 0, gacc])) * dt
angvel_new = angvel + torque * dt
quat_new = quat + (-0.5 * Sw * quat) * dt
mass_new = mass

pos_dot = pos_new.diff(dt)
vel_dot = vel_new.diff(dt)
angvel_dot = angvel_new.diff(dt)
quat_dot = quat_new.diff(dt)
mass_dot = mass_new.diff(dt)

##################################################
# Differential Definitions
##################################################

defs = (
        (pos.diff(t), pos_dot),
        (vel.diff(t), vel_dot),
        (angvel.diff(t), angvel_dot),
        (quat.diff(t), quat_dot),
        (mass.diff(t), mass_dot),
        # (speed.diff(t),     speed_dot),  # type:ignore
        # (torque,            torque_new),
        )

##################################################
# Define State & Input Arrays
##################################################
# v = Matrix(symbols("vel_x vel_y vel_z"))
# a = Matrix(symbols("acc_x acc_y acc_z"))
# g = symbols("gacc")
# aa = Matrix([a[0], a[1], a[2] + g])
# vv = v.dot(aa)**0.5

state = Matrix([
    pos,
    vel,
    angvel,
    quat,
    mass,
    DirectUpdate(gacc, gacc_new),
    # speed,
    DirectUpdate(speed, vel.norm()),  # type:ignore
    # DirectUpdate(mach, mach_new.subs(speed, Symbol("speed"))),  # type:ignore
    # DirectUpdate(alpha, alpha_new),  # type:ignore
    # DirectUpdate(phi, phi_new),  # type:ignore
    # cg,
    # Ixx,
    # Iyy,
    # Izz,
    # DirectUpdate(torque, torque_new),
    # DirectUpdate('q', My / Iyy),
    # DirectUpdate("mc", My_coef),
    # DirectUpdate("clmb", CLMB),
    # DirectUpdate("cnb", CNB),
    ])

input = Matrix([
    acc,
    # DirectUpdate(torque, torque_new),
    torque,
    ])

##################################################
# Define dynamics
##################################################

dynamics = Matrix(state.diff(t))

##################################################
# Build Model
##################################################

model = MissileGeneric().from_expression(dt,
                                         state,
                                         input,
                                         dynamics,
                                         modules=[atmosphere.modules],
                                         definitions=defs)
