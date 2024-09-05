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
from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl.Aero.AeroTableSymbolic import AeroTableSymbolic
from japl.Math.RotationSymbolic import ecef_to_lla_sym
from japl.Library.Earth.Earth import Earth

DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=8)



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

q_0 = Function("q_0", real=True)(t)
q_1 = Function("q_1", real=True)(t)
q_2 = Function("q_2", real=True)(t)
q_3 = Function("q_3", real=True)(t)

q_0_dot = Function("q_0_dot", real=True)(t)
q_1_dot = Function("q_1_dot", real=True)(t)
q_2_dot = Function("q_2_dot", real=True)(t)
q_3_dot = Function("q_3_dot", real=True)(t)

r_i_x = Function("r_i_x", real=True)(t)
r_i_y = Function("r_i_y", real=True)(t)
r_i_z = Function("r_i_z", real=True)(t)

v_i_x = Function("v_i_x", real=True)(t)
v_i_y = Function("v_i_y", real=True)(t)
v_i_z = Function("v_i_z", real=True)(t)

vel_mag_ecef = Function("vel_mag_ecef")(t)

a_b_x = Symbol("a_b_x", real=True)
a_b_y = Symbol("a_b_y", real=True)
a_b_z = Symbol("a_b_z", real=True)

# Aerodynamics force vector
f_b_A_x = Function("f_b_A_x", real=True)(t)
f_b_A_y = Function("f_b_A_y", real=True)(t)
f_b_A_z = Function("f_b_A_z", real=True)(t)
f_b_A = Matrix([f_b_A_x, f_b_A_y, f_b_A_z])

# ENU-frame vectors
r_enu_m = Matrix(symbols("r_e, r_n, r_u"))
v_enu_m = Matrix(symbols("v_e, v_n, v_u"))
a_enu_m = Matrix(symbols("a_e, a_n, a_u"))
alt = r_enu_m[2]

# Motor thrust force vector
# f_b_T_x = Function("f_b_T_x", real=True)(t)
# f_b_T_y = Function("f_b_T_y", real=True)(t)
# f_b_T_z = Function("f_b_T_z", real=True)(t)
# f_b_T = Matrix([f_b_T_x, f_b_T_y, f_b_T_z])

thrust = Function("thrust", real=True)(t)

q_m = Matrix([q_0, q_1, q_2, q_3])
r_i_m = Matrix([r_i_x, r_i_y, r_i_z])  # eci position
v_i_m = Matrix([v_i_x, v_i_y, v_i_z])  # eci velocity
a_b_m = Matrix([a_b_x, a_b_y, a_b_z])  # body acceleration
a_i_m = Matrix(symbols("a_i_x, a_i_y, a_i_z"))  # eci acceleration

# Earth grav acceleration
g_b_m_x = Function("g_b_m_x", real=True)(t)
g_b_m_y = Function("g_b_m_y", real=True)(t)
g_b_m_z = Function("g_b_m_z", real=True)(t)
g_b_m = Matrix([g_b_m_x, g_b_m_y, g_b_m_z])

# Angle-of-Attack & Sideslip-Angle
alpha = Function("alpha", real=True)(t)
alpha_dot = Function("alpha_dot", real=True)(t)
alpha_dot_dot = Function("alpha_dot_dot", real=True)(t)

beta = Function("beta", real=True)(t)
beta_dot = Function("beta_dot", real=True)(t)
beta_dot_dot = Function("beta_dot_dot", real=True)(t)

# angular velocities (roll, pitch, yaw)
p = Function("p", real=True)(t)
q = Function("q", real=True)(t)
r = Function("r", real=True)(t)

mass = Symbol("mass", real=True)
mach = Function("mach", real=True)(t)
C_s = Symbol("C_s", real=True)  # speed of sound


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
epsilon = 1e-17
V = v_e_e.norm() + epsilon

# (10) Mach number
M = V / C_s

# (11) Dynamic pressure
# rho = 1.293  # kg * m^-3
rho = atmosphere.density(alt)  # kg * m^-3
q_bar = 0.5 * rho * V**2

##################################################
# invert aerodynamics
##################################################

f_b_T = Matrix([0, 0, thrust])

# (6)
gacc = -9.81
g_i_m = Matrix([gacc, 0, 0])
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

##################################################
# Aerodynamics Inversion
##################################################

alpha_total = aerotable.inv_aerodynamics(
        thrust,  # type:ignore
        a_c_new,
        q_bar,
        mass,
        alpha,
        phi,
        M,
        alt,
        0.0,  # iota
        )

# (51)
alpha_c_new = atan2(tan(alpha_total), cos(phi_Ac))
beta_c_new = atan2(tan(alpha_total), sin(phi_Ac))

##################################################
# ECI to ENU convesion
##################################################

# eci0 = Matrix([Earth.radius_equatorial, 0, 0])
ecef0 = Matrix([Earth.radius_equatorial, 0, 0])
# ecef0 = C_eci_to_ecef * eci0
lla0 = ecef_to_lla_sym(ecef0)
lat0, lon0, alt0 = lla0
C_ecef_to_enu = Matrix([
    [-sin(lon0), cos(lon0), 0],  # type:ignore
    [-sin(lat0) * cos(lon0), -sin(lat0) * sin(lon0), cos(lat0)],  # type:ignore
    [cos(lat0) * cos(lon0), cos(lat0) * sin(lon0), sin(lat0)]  # type:ignore
    ])
r_enu_e_new = C_ecef_to_enu * (r_e_m - ecef0)
v_enu_e_new = C_ecef_to_enu * v_e_e
a_enu_e_new = C_ecef_to_enu * a_e_e

##################################################
# Quaternion regularization term:
# NOTE: incorporate quaternion normaliztion into
#       the dynamics. for integration processes
#       with less accuracy lambda must be increased
#       (i.e.) lambda = ~100 for euler-method.
#       compared to lambda = ~1e-3 for runge-kutta
##################################################

lam = 1e-3
q_m_norm = ((q_m.T * q_m)**0.5)[0]
q_m_dot_reg = q_m_dot - lam * (q_m_norm - 1.0) * q_m

##################################################
# Definitions
##################################################

defs = (
       (q_m.diff(t), q_m_dot_reg),
       (r_i_m.diff(t), v_i_m),
       (v_i_m.diff(t), v_i_m_dot),

       (p.diff(t), p_dot),
       # (q.diff(t), q_new.diff(t)),
       # (r.diff(t), r_new.diff(t)),

       (alpha_state.diff(t), alpha_state_dot),
       (beta_state.diff(t), beta_state_dot),

       (omega_n, 50),
       (zeta, 0.7),
       (K_phi, 1),
       (omega_p, 20),
       (phi_c, 0),
       (T_r, 0.5),
       (C_s, atmosphere.speed_of_sound(alt)),
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
# acc_b_x = Symbol("acc_b_x", real=True)
# acc_b_y = Symbol("acc_b_y", real=True)
# acc_b_z = Symbol("acc_b_z", real=True)
# st_a_b_m = Matrix([acc_b_x, acc_b_y, acc_b_z])
v_e_m = Matrix(symbols("v_e_x, v_e_y, v_e_z"))

state = Matrix([
    q_m,
    r_i_m,
    v_i_m,
    alpha_state,
    beta_state,
    p,
    # DirectUpdate(q, q_new),
    # DirectUpdate(r, r_new),
    mass,
    DirectUpdate(r_enu_m, r_enu_e_new),
    DirectUpdate(v_enu_m, v_enu_e_new),
    DirectUpdate(a_enu_m, a_enu_e_new),
    DirectUpdate(mach, M),
    DirectUpdate(vel_mag_ecef, V),
    DirectUpdate(v_e_m, v_e_e),
    ])

input = Matrix([
    DirectUpdate(alpha_c, alpha_c_new),
    DirectUpdate(beta_c, beta_c_new),
    a_c_y,
    a_c_z,
    f_b_A,
    # f_b_T,
    thrust,
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

modules = [atmosphere.modules,
           aerotable.modules]

model = MissileGenericMMD.from_expression(dt,
                                          state,
                                          input,
                                          dynamics,
                                          modules=modules,
                                          definitions=defs,
                                          use_multiprocess_build=True)


print("saving model...")
with open(f"{DIR}/mmd.pickle", 'wb') as f:
    dill.dump(model, f)
quit()
