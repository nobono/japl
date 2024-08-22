from sympy import Matrix, Symbol, symbols
from sympy import sign, rad, sqrt
from sympy import sin, cos
from sympy import atan, atan2, tan
from sympy import sec
from sympy import Float
from japl import Model
from sympy import Function
from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl.Library.Earth.EarthModelSymbolic import EarthModelSymbolic
# from japl.Aero.AeroTableSymbolic import AeroTableSymbolic
# from japl.Math import RotationSymbolic
# from japl.Math import VecSymbolic



class MissileGenericMMD(Model):
    pass


################################################
# Tables
################################################

atmosphere = AtmosphereSymbolic()

##################################################
# Momentless Missile Dynamics Model
##################################################

t = symbols("t")
dt = symbols("dt")

earth = EarthModelSymbolic()
omega_e = earth.omega

q_0 = Function("q_0", real=True)(t)  # type:ignore
q_1 = Function("q_1", real=True)(t)  # type:ignore
q_2 = Function("q_2", real=True)(t)  # type:ignore
q_3 = Function("q_3", real=True)(t)  # type:ignore

q_0_dot = Function("q_0_dot", real=True)(t)  # type:ignore
q_1_dot = Function("q_1_dot", real=True)(t)  # type:ignore
q_2_dot = Function("q_2_dot", real=True)(t)  # type:ignore
q_3_dot = Function("q_3_dot", real=True)(t)  # type:ignore

pos_i_x = Function("pos_i_x", real=True)(t)  # type:ignore
pos_i_y = Function("pos_i_y", real=True)(t)  # type:ignore
pos_i_z = Function("pos_i_z", real=True)(t)  # type:ignore

vel_i_x = Function("vel_i_x", real=True)(t)  # type:ignore
vel_i_y = Function("vel_i_y", real=True)(t)  # type:ignore
vel_i_z = Function("vel_i_z", real=True)(t)  # type:ignore

acc_b_x = Symbol("acc_b_x", real=True)  # type:ignore
acc_b_y = Symbol("acc_b_y", real=True)  # type:ignore
acc_b_z = Symbol("acc_b_z", real=True)  # type:ignore

# Aerodynamics force vector
f_b_A_x = Function("f_b_A_x", real=True)(t)  # type:ignore
f_b_A_y = Function("f_b_A_y", real=True)(t)  # type:ignore
f_b_A_z = Function("f_b_A_z", real=True)(t)  # type:ignore
f_b_A = Matrix([f_b_A_x, f_b_A_y, f_b_A_z])

# Motor thrust force vector
f_b_T_x = Function("f_b_T_x", real=True)(t)  # type:ignore
f_b_T_y = Function("f_b_T_y", real=True)(t)  # type:ignore
f_b_T_z = Function("f_b_T_z", real=True)(t)  # type:ignore
f_b_T = Matrix([f_b_T_x, f_b_T_y, f_b_T_z])

q_m = Matrix([q_0, q_1, q_2, q_3])
r_i_m = Matrix([pos_i_x, pos_i_y, pos_i_z])  # eci position
v_i_m = Matrix([vel_i_x, vel_i_y, vel_i_z])  # eci velocity
a_b_m = Matrix([acc_b_x, acc_b_y, acc_b_z])  # body acceleration

mass = Symbol("mass", real=True)

C_s = Symbol("C_s", real=True)  # speed of sound

# Earth grav acceleration
g_b_m_x = Function("g_b_m_x", real=True)(t)  # type:ignore
g_b_m_y = Function("g_b_m_y", real=True)(t)  # type:ignore
g_b_m_z = Function("g_b_m_z", real=True)(t)  # type:ignore
g_b_m = Matrix([g_b_m_x, g_b_m_y, g_b_m_z])

# Angle-of-Attack & Sideslip-Angle
alpha, alpha_dot, alpha_dot_dot = symbols("alpha, alpha_dot, alpha_dot_dot", real=True)
beta, beta_dot, beta_dot_dot = symbols("beta, beta_dot, beta_dot_dot", real=True)

##################################################
# 2.1 ECI Position and Velocity Derivatives
##################################################

# (1)
C_eci_to_ecef = Matrix([
    [cos(omega_e * t), sin(omega_e * t), 0],
    [-sin(omega_e * t), cos(omega_e * t), 0],  # type:ignore
    [0, 0, 1]])

# (4)
C_body_to_eci = Matrix([
    [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],   # type:ignore # noqa
    [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],   # type:ignore # noqa
    [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]])  # type:ignore # noqa

# (6)
a_b_m = ((f_b_A + f_b_T) / mass) + g_b_m

# (2)
r_i_m_dot = v_i_m

# (3)
v_i_m_dot = C_body_to_eci * a_b_m

# (7) Earth-relative position vector
r_e_m = C_eci_to_ecef * r_i_m

# (9)
omega_skew_ie = Matrix([
    [0, -omega_e, 0],
    [omega_e, 0, 0],
    [0, 0, 0],
    ])

# (8) Earth-relative velocity vector
v_e_e = C_eci_to_ecef * v_i_m - omega_skew_ie * r_e_m

# (12)
V = v_e_e.norm()

# (10) Mach number
M = V / C_s

# (11) Dynamic pressure
# TODO: fix rho
# rho = atmosphere.density(alt)
rho = 1.293  # kg * m^-3
q_bar = 0.5 * rho * V**2

# (14)
C_body_to_ecef = C_eci_to_ecef * C_body_to_eci

# (13) Earth-relative acceleration vector
a_e_e = C_body_to_ecef * a_b_m - (2 * omega_skew_ie * v_e_e)\
        - (omega_skew_ie * omega_skew_ie * r_e_m)

##################################################
# 2.2 MMD Autopilot Transfer Functions
##################################################
omega_n = Symbol("omega_n", real=True)  # natural frequency
zeta = Symbol("zeta", real=True)  # damping ratio

# (16) Angle of attack transfer function
alpha_c = Symbol("alpha_c", real=True)  # angle of attack command
alpha_state = Matrix([alpha, alpha_dot])
A_alpha = Matrix([
    [0, 1],
    [-omega_n**2, -2 * zeta * omega_n]  # type:ignore
    ])
B_alpha = Matrix([
    [0],
    [omega_n**2]
    ])
alpha_state_dot = A_alpha * alpha_state + B_alpha * alpha_c

# (17) Sideslip angle transfer function
beta_c = Symbol("beta_c", real=True)  # sideslip angle command
beta_state = Matrix([beta, beta_dot])
A_beta = Matrix([
    [0, 1],
    [-omega_n**2, -2 * zeta * omega_n]  # type:ignore
    ])
B_beta = Matrix([
    [0],
    [omega_n**2]
    ])
beta_state_dot = A_beta * beta_state + B_beta * beta_c

############
# Roll
############

# (18) Skid-to_Turn (STT) roll angular velocity
p, q, r = symbols("p, q, r", real=True)  # angular velocities (roll, pitch, yaw)
K_phi = Symbol("K_phi", real=True)       # roll gain
omega_p = Symbol("omega_p", real=True)   # natural frequency (roll)
phi_c = Symbol("phi_c", real=True)       # roll angle command

# (19) Pseudo-roll angle
phi_hat, phi_hat_dot = symbols("phi_hat, phi_hat_dot", real=True)
phi_hat_dot = p

# (23)
C_eci_to_body = C_body_to_eci.T
phi = atan2(C_eci_to_body[1, 2], C_eci_to_body[2, 2])

# (24)
T_r = Symbol("T_r", real=True)  # roll autopilot time constant
phi_hat = (1 / T_r) * (phi_c - phi)  # type:ignore

# (22)
phi_hat_c = phi_c - phi  # type:ignore

# (21)
p_c = K_phi * (phi_hat_c - phi_hat)

# (20)
p_dot = omega_p * (p_c - p)

##################################################
# 2.3 Quaternion State Derivatives
##################################################

# (26)
omega_b_ib = Matrix([p, q, r])

# (25)
Sq = Matrix([[-q_1, -q_2, -q_3],    # type:ignore
             [q_0, -q_3, q_2],      # type:ignore
             [q_3, q_0, -q_1],      # type:ignore
             [-q_2, q_1, q_0]])     # type:ignore
q_m_dot = 0.5 * Sq * omega_b_ib

##################################################
# 2.4 Computation of the Missile Angular Velocity
##################################################

###############################
# NOTE: Earth-relative velocity
# must be derived as a function
# of MMD states:
#   - alpha
#   - beta
#   - alpha_dot
#   - beta_dot

# (29)
C_ecef_to_body = C_body_to_ecef.T
a_b_e = C_ecef_to_body * a_e_e

# (32) (33) (34)
u_hat = (1 + tan(alpha)**2 + tan(beta)**2)  # type:ignore
v_hat = u_hat * tan(beta)
w_hat = u_hat * tan(alpha)

# (30)
v_b_e_hat = Matrix([u_hat, v_hat, w_hat])

# (35) (36) (37)
u_hat_dot = -(1 + tan(alpha)**2 + tan(beta)**2)**(-3 / 2) * ((alpha_dot * tan(alpha) * sec(alpha)**2) + (beta_dot * tan(beta) * sec(beta)**2)) # type:ignore # noqa

# (31)
v_b_e_hat_dot = Matrix([u_hat_dot, v_hat_dot, w_hat_dot])

# (39)
v_b_e = V * v_b_e_hat
u = v_b_e[0]
v = v_b_e[1]
w = v_b_e[2]

# (38)
V_dot = a_b_e * v_b_e_hat

# (40)
v_b_e_dot = V_dot * v_b_e_hat + V * v_b_e_hat_dot
u_dot = v_b_e_dot[0]
v_dot = v_b_e_dot[1]
w_dot = v_b_e_dot[2]

###############################

# (28)
omega_skew_ib = Matrix([
    [0, -r, q],
    [r, 0, -q],
    [-q, p, 0]
    ])

# (27)
a_b_e = v_b_e_dot + (omega_skew_ib - C_ecef_to_body * omega_skew_ie * C_body_to_ecef) * v_b_e

# (41) C_ij are the array elements of C^b_e
C_11 = C_body_to_ecef[0, 0]
C_12 = C_body_to_ecef[0, 1]
C_13 = C_body_to_ecef[0, 2]
C_21 = C_body_to_ecef[1, 0]
C_22 = C_body_to_ecef[1, 1]
C_23 = C_body_to_ecef[1, 2]
C_31 = C_body_to_ecef[2, 0]
C_32 = C_body_to_ecef[2, 1]
C_33 = C_body_to_ecef[2, 2]

# (44)
C_1 = (C_11 * C_32 - C_12 * C_31) * u + (C_21 * C_32 - C_22 * C_31) * v
C_2 = (C_11 * C_22 - C_12 * C_21) * u + (C_22 * C_31 - C_21 * C_32) * w

# (42) (43)
q = (w_dot - a_b_e[2] + p * v - omega_e * C_1) / u
r = (a_b_e[1] - v_dot + p * w + omega_e * C_2) / u

# (46) NOTE: this is for a BTT missile

##################################################
# 3.1 Tail-Control Guidance Command Mapping
##################################################

# autopilot acceleration commands
a_c_x, a_c_y, a_c_z = symbols("a_c_x, a_c_y, a_c_z", real=True)
a_c = Matrix([a_c_x, a_c_y, a_c_z])

# (48) Total life coefficient command
Sref = Symbol("Sref", real=True)
C_N_c = (a_c * mass) / q_bar * Sref

# (49) Aerodynamics roll angle command
phi_Ac = atan2(-a_c_y, -a_c_z)

# (50)
a_c = sqrt(a_c_y**2 + a_c_z**2)

# (51)


quit()
##################################################
# States
##################################################

pos = Matrix([pos_x, pos_y, pos_z])
vel = Matrix([vel_x, vel_y, vel_z])
angvel = Matrix([angvel_x, angvel_y, angvel_z])
quat = Matrix([q_0, q_1, q_2, q_3])
mass = symbols("mass")
speed = Symbol("speed", real=True)  # type:ignore
mach = Function("mach", real=True)(t)  # type:ignore

alpha = Symbol("alpha", real=True)  # type:ignore
phi = Symbol("phi", real=True)  # type:ignore
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
gravity = Matrix([0, 0, gacc])

##################################################
# AeroTable
##################################################

# alt = pos[2]
# gacc_new = atmosphere.grav_accel(alt)  # type:ignore
# sos = atmosphere.speed_of_sound(alt)  # type:ignore
# speed_new = vel.norm()
# mach_new = speed_new / sos  # type:ignore

# # calc angle of attack: (pitch_angle - flight_path_angle)
# vel_hat = vel / speed   # flight path vector

# # projection vel_hat --> x-axis
# zx_plane_norm = Matrix([0, 1, 0])
# vel_hat_zx = ((vel_hat.dot(zx_plane_norm)) / zx_plane_norm.norm()) * zx_plane_norm
# vel_hat_proj = vel_hat - vel_hat_zx

# # get Trait-bryan angles (yaw, pitch, roll)
# yaw_angle, pitch_angle, roll_angle = RotationSymbolic.quat_to_tait_bryan_sym(quat)

# # angle between proj vel_hat & xaxis
# x_axis_inertial = Matrix([1, 0, 0])
# ang = VecSymbolic.vec_ang_sym(vel_hat_proj, x_axis_inertial)

# flight_path_angle = sign(vel_hat_proj[2]) * ang  # type:ignore
# alpha_new = pitch_angle - flight_path_angle  # angle of attack
# phi_new = roll_angle

# iota = rad(0.1)
# CLMB = -aerotable.get_CLMB_Total(alpha, phi, mach, iota)  # type:ignore
# CNB = aerotable.get_CNB_Total(alpha, phi, mach, iota)  # type:ignore
# My_coef = CLMB + (cg - aerotable.get_MRC()) * CNB  # type:ignore

# q = atmosphere.dynamic_pressure(vel, pos_z)  # type:ignore
# Sref = aerotable.get_Sref()
# Lref = aerotable.get_Lref()
# My = My_coef * q * Sref * Lref

# force_z_aero = CNB * q * Sref  # type:ignore
# force_aero = Matrix([0, 0, force_z_aero])
# force_new = force + force_aero

# torque_y_aero = My / Iyy
# torque_aero = Matrix([0, torque_y_aero, 0])
# torque_new = torque + torque_aero

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
        (pos.diff(t), pos_dot),
        (vel.diff(t), vel_dot),
        (angvel.diff(t), angvel_dot),
        (quat.diff(t), quat_dot),
        (mass.diff(t), mass_dot),
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
    DirectUpdate(gacc, gacc_new),  # type:ignore
    DirectUpdate(speed, speed_new),
    DirectUpdate(mach, mach_new),  # type:ignore
    DirectUpdate(alpha, alpha_new),
    DirectUpdate(phi, phi_new),
    ])

input = Matrix([
    DirectUpdate(force, force_new),
    DirectUpdate(torque, torque_new),
    ])

##################################################
# Define dynamics
##################################################

dynamics = state.diff(t)

##################################################
# Build Model
##################################################

model = MissileGenericMMD.from_expression(dt,
                                          state,
                                          input,
                                          dynamics,
                                          modules=[atmosphere.modules,
                                                   aerotable.modules],
                                          definitions=defs)
