import dill
import os
import numpy as np
from sympy import Matrix, Symbol, symbols
from sympy import sign, rad, sqrt
from sympy import sin, cos
from sympy import atan, atan2, tan
from sympy import sec
from sympy import Float
import sympy as sp
from japl import Model
from sympy import Function
from japl import Sim, SimObject
from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl.BuildTools.BuildTools import build_model
from japl.Library.Earth.EarthModelSymbolic import EarthModelSymbolic
from japl.Aero.AeroTableSymbolic import AeroTableSymbolic
from japl.Math.Rotation import ecef_to_lla
from japl.Math.Rotation import ecef_to_enu
from japl.Math.Rotation import ecef_to_eci
from japl.Math.Rotation import eci_to_ecef
from japl.Math.Rotation import eci_to_enu
from japl.Math.RotationSymbolic import ecef_to_lla_sym
from japl.Library.Earth.Earth import Earth
from japl.Util.Desym import Desym
from japl import PyQtGraphPlotter

DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=8)

# from japl.Math import RotationSymbolic
# from japl.Math import VecSymbolic
# earth = EarthModelSymbolic()
# lla0 = np.array([0, 0, 0])
# ecef0 = [earth.radius_equatorial, 0, 0]
# t = 1
# eci = np.array([earth.radius_equatorial, 0, 0])
# ecef = eci_to_ecef(eci, t=t)
# enu = ecef_to_enu(ecef, ecef0)
# print(enu)
# print(eci_to_enu(eci, ecef0, t=t))
# quit()



class MissileGenericMMD(Model):
    pass


################################################
# Tables
################################################

atmosphere = AtmosphereSymbolic()
aerotable = AeroTableSymbolic(DIR + "/../../../aeromodel/cms_sr_stage1aero.mat",
                              from_template="CMS")

##################################################
# Momentless Missile Dynamics Model
##################################################

t = Symbol("t")
dt = Symbol("dt")

omega_e = Earth.omega

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

thrust = Function("thrust", real=True)(t)    # type:ignore

q_m = Matrix([q_0, q_1, q_2, q_3])
r_i_m = Matrix([pos_i_x, pos_i_y, pos_i_z])  # eci position
v_i_m = Matrix([vel_i_x, vel_i_y, vel_i_z])  # eci velocity
a_b_m = Matrix([acc_b_x, acc_b_y, acc_b_z])  # body acceleration
a_i_m = Matrix(symbols("acc_i_x, acc_i_y, acc_i_z"))  # eci acceleration

mass = Symbol("mass", real=True)

mach = Function("mach", real=True)(t)  # type:ignore

C_s = Symbol("C_s", real=True)  # speed of sound

# Earth grav acceleration
g_b_m_x = Function("g_b_m_x", real=True)(t)  # type:ignore
g_b_m_y = Function("g_b_m_y", real=True)(t)  # type:ignore
g_b_m_z = Function("g_b_m_z", real=True)(t)  # type:ignore
g_b_m = Matrix([g_b_m_x, g_b_m_y, g_b_m_z])

# Angle-of-Attack & Sideslip-Angle
alpha = Function("alpha", real=True)(t)  # type:ignore
alpha_dot = Function("alpha_dot", real=True)(t)  # type:ignore
alpha_dot_dot = Function("alpha_dot_dot", real=True)(t)  # type:ignore

beta = Function("beta", real=True)(t)  # type:ignore
beta_dot = Function("beta_dot", real=True)(t)  # type:ignore
beta_dot_dot = Function("beta_dot_dot", real=True)(t)  # type:ignore

ecef_to_lla_map = Function("ecef_to_lla_map")
ecef_to_enu_map = Function("ecef_to_enu_map")
ecef_to_eci_map = Function("ecef_to_eci_map")
eci_to_ecef_map = Function("eci_to_ecef_map")
eci_to_enu_map = Function("eci_to_enu_map")

##################################################
# 2.1 ECI Position and Velocity Derivatives
##################################################

# (1)
C_eci_to_ecef = Matrix([
    [cos(omega_e * t), sin(omega_e * t), 0],   # type:ignore
    [-sin(omega_e * t), cos(omega_e * t), 0],  # type:ignore
    [0, 0, 1]])

# (4)
C_body_to_eci = Matrix([
    [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],   # type:ignore # noqa
    [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],   # type:ignore # noqa
    [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa

# (14)
C_body_to_ecef = C_eci_to_ecef * C_body_to_eci

##################################################

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

##################################################

# (12)
# V = v_e_e.norm()
V = ((v_e_e.T * v_e_e)**0.5)[0]

# (10) Mach number
M = V / C_s

# (11) Dynamic pressure
# TODO: fix rho
# rho = atmosphere.density(alt)
rho = 1.293  # kg * m^-3
q_bar = 0.5 * rho * V**2

##################################################
# invert aerodynamics
##################################################

# (6)
g_i_m = Matrix([-9.81, 0, 0])
g_b_m = C_body_to_eci.T * g_i_m
a_b_m = ((f_b_A + f_b_T) / mass) + g_b_m

# (13) Earth-relative acceleration vector
a_e_e = C_body_to_ecef * a_b_m - (2 * omega_skew_ie * v_e_e)\
        - (omega_skew_ie * omega_skew_ie * r_e_m)

# (29)
C_ecef_to_body = C_body_to_ecef.T
a_b_e = C_ecef_to_body * a_e_e

##################################################

# (32) (33) (34)
u_hat = (1 + tan(alpha)**2 + tan(beta)**2)  # type:ignore
v_hat = u_hat * tan(beta)
w_hat = u_hat * tan(alpha)

# (35) (36) (37)
u_hat_dot = -(1 + tan(alpha)**2 + tan(beta)**2)**(-3 / 2) * ((alpha_dot * tan(alpha) * sec(alpha)**2) + (beta_dot * tan(beta) * sec(beta)**2)) # type:ignore # noqa
v_hat_dot = u_hat_dot * tan(beta) + u_hat * beta_dot * sec(beta)**2  # type:ignore
w_hat_dot = u_hat_dot * tan(alpha) + u_hat * alpha_dot * sec(alpha)**2  # type:ignore

# (30)
v_b_e_hat = Matrix([u_hat, v_hat, w_hat])

# (31)
v_b_e_hat_dot = Matrix([u_hat_dot, v_hat_dot, w_hat_dot])

##################################################

# p, q, r = symbols("p, q, r", real=True)  # angular velocities (roll, pitch, yaw)
p = Function("p", real=True)(t)  # type:ignore
q = Function("q", real=True)(t)  # type:ignore
r = Function("r", real=True)(t)  # type:ignore

###############################
# NOTE: Earth-relative velocity
# must be derived as a function
# of MMD states:
#   - alpha
#   - beta
#   - alpha_dot
#   - beta_dot

# (39)
v_b_e = V * v_b_e_hat
u = v_b_e[0]
v = v_b_e[1]
w = v_b_e[2]

# (38)
V_dot = a_b_e.T * v_b_e_hat

# (40)
v_b_e_dot = V_dot[0] * v_b_e_hat + V * v_b_e_hat_dot
u_dot = v_b_e_dot[0]
v_dot = v_b_e_dot[1]
w_dot = v_b_e_dot[2]

# (41) C_ij are the array elements of C^b_e
C_11 = C_body_to_ecef.T[0, 0]
C_12 = C_body_to_ecef.T[0, 1]
C_13 = C_body_to_ecef.T[0, 2]
C_21 = C_body_to_ecef.T[1, 0]
C_22 = C_body_to_ecef.T[1, 1]
C_23 = C_body_to_ecef.T[1, 2]
C_31 = C_body_to_ecef.T[2, 0]
C_32 = C_body_to_ecef.T[2, 1]
C_33 = C_body_to_ecef.T[2, 2]

# (44)
C_1 = (C_11 * C_32 - C_12 * C_31) * u + (C_21 * C_32 - C_22 * C_31) * v
C_2 = (C_11 * C_22 - C_12 * C_21) * u + (C_22 * C_31 - C_21 * C_32) * w

# (42) (43)
q_new = (w_dot - a_b_e[2] + p * v - omega_e * C_1) / u
r_new = (a_b_e[1] - v_dot + p * w + omega_e * C_2) / u

##################################################

# (26)
# TODO
# TODO
# TODO q -> q_new?
omega_b_ib = Matrix([p, q_new, r_new])

##################################################

# (2)
r_i_m_dot = v_i_m

# (3)
v_i_m_dot = C_body_to_eci * a_b_m

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
K_phi = Symbol("K_phi", real=True)       # roll gain
omega_p = Symbol("omega_p", real=True)   # natural frequency (roll)
phi_c = Symbol("phi_c", real=True)       # roll angle command
T_r = Symbol("T_r", real=True)           # roll autopilot time constant

# (19) Pseudo-roll angle
phi_hat, phi_hat_dot = symbols("phi_hat, phi_hat_dot", real=True)
phi_hat_dot = p

# (23)
C_eci_to_body = C_body_to_eci.T
phi = atan2(C_eci_to_body[1, 2], C_eci_to_body[2, 2])

# (24)
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

# (28)
omega_skew_ib = Matrix([
    [0, -r, q],  # type:ignore
    [r, 0, -q],  # type:ignore
    [-q, p, 0]   # type:ignore
    ])

# (27)
# TODO: include this or not?
# a_b_e = v_b_e_dot + (omega_skew_ib - C_ecef_to_body * omega_skew_ie * C_body_to_ecef) * v_b_e


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
a_c_new = sqrt(a_c_y**2 + a_c_z**2)

###########################################
r_enu_m = Matrix(symbols("r_e, r_n, r_u"))
v_enu_m = Matrix(symbols("v_e, v_n, v_u"))
a_enu_m = Matrix(symbols("a_e, a_n, a_u"))

eci0 = Matrix([Earth.radius_equatorial, 0, 0])
ecef0 = C_eci_to_ecef * eci0
lla0 = ecef_to_lla_sym(ecef0)
lat, lon, alt = lla0

C_ecef_to_enu = Matrix([
    [-sin(lon), cos(lon), 0],  # type:ignore
    [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],  # type:ignore
    [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]  # type:ignore
    ])

C_eci_to_enu = C_eci_to_ecef * C_ecef_to_enu

r_ecef_m = C_eci_to_ecef * r_i_m
r_enu_m_new = C_ecef_to_enu * (r_ecef_m - ecef0)

v_enu_m_new = C_eci_to_enu * v_i_m
a_enu_m_new = C_body_to_eci * C_eci_to_enu * a_b_m


alt = r_enu_m[2]
alpha_total = aerotable.inv_aerodynamics(
        thrust,  # type:ignore
        a_c_new,
        q_bar,
        mass,
        alpha,
        phi,
        M,
        alt,
        0.0,
        )

#############################################
# thrust = Symbol("thrust")
# acc_cmd = Symbol("acc_cmd")
# dynq = Symbol("dynq")
# mass = Symbol("mass")
# alpha = Symbol("alpha")
# phi = Symbol("phi")
# mach = Symbol("mach")
# alt = Symbol("alt")
# iota = Symbol("iota")

# _vars = (thrust,
#          acc_cmd,
#          dynq,
#          mass,
#          alpha,
#          phi,
#          mach,
#          alt,
#          iota)

# _func = aerotable.inv_aerodynamics(thrust,
#                                    acc_cmd,
#                                    dynq,
#                                    mass,
#                                    alpha,
#                                    phi,
#                                    mach,
#                                    alt,
#                                    iota)
# f = Desym(_vars, _func, modules=[aerotable.modules], dummify=True)

# thrust = 50_000.0
# am_c = 4.848711297583447
# q_bar = 30.953815676156566
# mass = 3.040822554083198e+02
# alpha = np.radians(0.021285852300486)
# mach = 0.020890665777000
# alt = 0.097541062161326
# ret = f(thrust, am_c, q_bar, mass, alpha, 0, mach, alt, 0)
# print(ret)
# quit()
#############################################

# (51)
alpha_c_out = atan2(tan(alpha_total), cos(phi_Ac))
beta_c_out = atan2(tan(alpha_total), sin(phi_Ac))

##################################################
# Differential Definitions
##################################################
alt = r_enu_m_new[2]
# gacc = atmosphere.grav_accel(alt)  # type:ignore
gacc = -9.81
g_b_m_new = (r_i_m / r_i_m.norm()) * gacc

# CA_Boost = aerotable.get_CA_Boost(alpha, phi, M, alt)  # type:ignore
# CNB = aerotable.get_CNB(alpha, phi, M)  # type:ignore
# q_bar_new = atmosphere.dynamic_pressure(v_b_m, alt)  # type:ignore
# Sref = aerotable.get_Sref()
# Lref = aerotable.get_Lref()
# force_z_aero = CNB * q_bar_new * Sref  # type:ignore
# force_y_aero = CA_Boost * q_bar_new * Sref  # type:ignore
# force_aero = Matrix([0, force_y_aero, force_z_aero])


##################################################
# add quaternion regularization term
# NOTE: how to incorporate quaternion normaliztion
#       into the dynamics. for integration processes
#       with less accuracy lambda must be increased
#       (i.e.) lambda = ~100 for euler-method.
#       compared to lambda = ~1e-3 for runge-kutta
##################################################
lam = 1e-3
q_m_norm = ((q_m.T * q_m)**0.5)[0]
q_m_dot_2 = q_m_dot - lam * (q_m_norm - 1.0) * q_m
##################################################

vel_mag = Function("vel_mag")(t)

defs = (
       (q_m.diff(t), q_m_dot_2),
       (r_i_m.diff(t), v_i_m),
       (v_i_m.diff(t), C_body_to_eci * a_b_m),
       (p.diff(t), p_dot),  # type:ignore
       # (q, q_new),
       # (r, r_new),
       (alpha_state.diff(t), alpha_state_dot),
       (beta_state.diff(t), beta_state_dot),
       (omega_n, sp.Float(50)),
       (zeta, sp.Float(0.7)),

       (K_phi, sp.Float(1)),
       (omega_p, sp.Float(20)),
       (phi_c, sp.Float(0)),
       (T_r, sp.Float(0.5)),
       (C_s, 343),

       # (vel_mag.diff(t), V.diff(t)),
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
acc_b_x = Symbol("acc_b_x", real=True)  # type:ignore
acc_b_y = Symbol("acc_b_y", real=True)  # type:ignore
acc_b_z = Symbol("acc_b_z", real=True)  # type:ignore
st_a_b_m = Matrix([acc_b_x, acc_b_y, acc_b_z])


state = Matrix([
    q_m,
    r_i_m,
    v_i_m,
    alpha_state,
    beta_state,
    p,
    # q,
    # r,
    mass,
    DirectUpdate(r_enu_m, r_enu_m_new),
    DirectUpdate(v_enu_m, v_enu_m_new),
    DirectUpdate(a_enu_m, a_enu_m_new),
    # DirectUpdate(a_i_m, C_eci_to_ecef.T * a_e_e),
    DirectUpdate(mach, M),
    DirectUpdate(vel_mag, V),
    # vel_mag,
    ])

input = Matrix([
    # DirectUpdate(f_b_A, force_aero),
    alpha_c,
    beta_c,
    f_b_A,
    f_b_T,
    ])

##################################################
# Define dynamics
##################################################

# state.subs(b4subs)

dynamics = state.diff(t)

# for i, j in zip(q_m.diff(t), q_m_dot):
#     dynamics[0].subs(i, j)
#     dynamics[1].subs(i, j)
#     dynamics[2].subs(i, j)
#     dynamics[3].subs(i, j)

# quit()

##################################################
# Build Model
##################################################
# (state_vars,
#  input_vars,
#  dynamics_expr) = build_model(Matrix(state),
#                               Matrix(input),
#                               Matrix(dynamics),
#                               defs)

custom_modules = {"ecef_to_lla_map": ecef_to_lla,
                  "ecef_to_enu_map": ecef_to_enu,
                  "ecef_to_eci_map": ecef_to_eci,
                  "eci_to_ecef_map": eci_to_ecef,
                  "eci_to_enu_map": eci_to_enu}

modules = [atmosphere.modules,
           aerotable.modules,
           custom_modules]

model = MissileGenericMMD.from_expression(dt,
                                          state,
                                          input,
                                          dynamics,
                                          modules=modules,
                                          definitions=defs)


print("saving model...")
with open(f"{DIR}/mmd.pickle", 'wb') as f:
    dill.dump(model, f)
quit()
