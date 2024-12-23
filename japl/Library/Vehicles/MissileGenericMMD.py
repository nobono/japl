import os
import numpy as np
from sympy import Matrix, Symbol, symbols
from sympy import Piecewise
from sympy import sign, rad, sqrt
from sympy import sin, cos
from sympy import atan, atan2, tan
from sympy import sec
from sympy import Float
import sympy as sp
from sympy import Abs
from japl import Model
from sympy import Function
from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl.Aero.AeroTableSymbolic import AeroTableSymbolic
from japl.Math.RotationSymbolic import ecef_to_lla_sym
from japl.Library.Earth.Earth import Earth
from japl.Util.Util import flatten_list
from japl.BuildTools.BuildTools import to_pycode
from japl import JAPL_HOME_DIR

DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=8)



class MissileGenericMMD(Model):
    pass


model = MissileGenericMMD()


################################################
# Tables
################################################

atmosphere = AtmosphereSymbolic()
aerotable = AeroTableSymbolic(JAPL_HOME_DIR + "/aeromodel/stage_1_aero.mat",
                              from_template="orion")
# aerotable = AeroTableSymbolic()

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

# ECI-frame
r_i_x = Function("r_i_x", real=True)(t)
r_i_y = Function("r_i_y", real=True)(t)
r_i_z = Function("r_i_z", real=True)(t)

v_i_x = Function("v_i_x", real=True)(t)
v_i_y = Function("v_i_y", real=True)(t)
v_i_z = Function("v_i_z", real=True)(t)

a_i_x = Function("a_i_x", real=True)(t)
a_i_y = Function("a_i_y", real=True)(t)
a_i_z = Function("a_i_z", real=True)(t)

r_i_m = Matrix([r_i_x, r_i_y, r_i_z])  # eci position
v_i_m = Matrix([v_i_x, v_i_y, v_i_z])  # eci velocity
a_i_m = Matrix([a_i_x, a_i_y, a_i_z])  # eci acceleration

# ECEF-frame
r_e_x = Function("r_e_x", real=True)(t)
r_e_y = Function("r_e_y", real=True)(t)
r_e_z = Function("r_e_z", real=True)(t)

v_e_x = Function("v_e_x", real=True)(t)
v_e_y = Function("v_e_y", real=True)(t)
v_e_z = Function("v_e_z", real=True)(t)

a_e_x = Function("a_e_x", real=True)(t)
a_e_y = Function("a_e_y", real=True)(t)
a_e_z = Function("a_e_z", real=True)(t)

r_e_m = Matrix([r_e_x, r_e_y, r_e_z])
v_e_m = Matrix([v_e_x, v_e_y, v_e_z])
a_e_m = Matrix([a_e_x, a_e_y, a_e_z])

vel_mag_e = Function("vel_mag_e")(t)
vel_mag_e_dot = Function("vel_mag_e_dot")(t)

# BODY-frame
v_b_e_x = Function("v_b_e_x", real=True)(t)
v_b_e_y = Function("v_b_e_y", real=True)(t)
v_b_e_z = Function("v_b_e_z", real=True)(t)
v_b_e_m = Matrix([v_b_e_x, v_b_e_y, v_b_e_z])

v_b_e_hat_x = Function("v_b_e_hat_x", real=True)(t)
v_b_e_hat_y = Function("v_b_e_hat_y", real=True)(t)
v_b_e_hat_z = Function("v_b_e_hat_z", real=True)(t)
v_b_e_m_hat = Matrix([v_b_e_hat_x, v_b_e_hat_y, v_b_e_hat_z])

a_b_x = Symbol("a_b_x", real=True)
a_b_y = Symbol("a_b_y", real=True)
a_b_z = Symbol("a_b_z", real=True)

# Aerodynamics force vector
f_b_A_x = Function("f_b_A_x", real=True)(t)
f_b_A_y = Function("f_b_A_y", real=True)(t)
f_b_A_z = Function("f_b_A_z", real=True)(t)
f_b_A = Matrix([f_b_A_x, f_b_A_y, f_b_A_z])

lift = Symbol("lift", real=True)
slip = Symbol("slip", real=True)
drag = Symbol("drag", real=True)

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
a_b_e_m = Matrix([a_b_x, a_b_y, a_b_z])  # body acceleration

# Earth grav acceleration
g_b_m_x = Function("g_b_m_x", real=True)(t)
g_b_m_y = Function("g_b_m_y", real=True)(t)
g_b_m_z = Function("g_b_m_z", real=True)(t)
g_b_m = Matrix([g_b_m_x, g_b_m_y, g_b_m_z])

# Angle-of-Attack
alpha = Function("alpha", real=True)(t)
alpha_dot = Function("alpha_dot", real=True)(t)
alpha_dot_dot = Function("alpha_dot_dot", real=True)(t)

# Sideslip-Angle
beta = Function("beta", real=True)(t)
beta_dot = Function("beta_dot", real=True)(t)
beta_dot_dot = Function("beta_dot_dot", real=True)(t)

# Roll-Angle
# phi = Function("phi", real=True)(t)
phi_hat = Function("phi_hat", real=True)(t)     # Pseudo-roll angle
phi_hat_dot = Symbol("phi_hat_dot", real=True)  # Pseudo-roll angle rate

# Angle Commands
alpha_c = Symbol("alpha_c", real=True)          # angle of attack command
beta_c = Symbol("beta_c", real=True)            # sideslip angle command
phi_c = Symbol("phi_c", real=True)              # roll angle command

# Autopilot transfer function terms
omega_n = Symbol("omega_n", real=True)          # alpha & beta natural frequency
zeta = Symbol("zeta", real=True)                # alpha & beta damping ratio
omega_p = Symbol("omega_p", real=True)          # roll natural frequency
T_r = Symbol("T_r", real=True)                  # roll autopilot time constant
K_phi = Symbol("K_phi", real=True)              # roll controller gain

is_boosting = Symbol("is_boosting", real=True)  # vehicle is boosting
stage = Symbol("stage")                         # missile stage (int)
is_launched = Symbol("is_launched", real=True)  # flag to launch missile or keep stationary

# angular velocities
p = Function("p", real=True)(t)                 # roll-rate
q = Function("q", real=True)(t)                 # pitch-rate
r = Function("r", real=True)(t)                 # yaw-rate

wet_mass = Function("wet_mass", real=True)(t)
dry_mass = Symbol("dry_mass", real=True)
mass_dot = Symbol("mass_dot", real=True)
Sref = Symbol("Sref", real=True)                # area reference

mach = Function("mach", real=True)(t)           # mach number
C_s = Symbol("C_s", real=True)                  # speed of sound
rho = Symbol("rho", real=True)                  # air density
gacc = Symbol("gacc", real=True)

# autopilot acceleration commands
a_c_x = Symbol("a_c_x", real=True)              # acc-x command (body-frame)
a_c_y = Symbol("a_c_y", real=True)              # acc-y command (body-frame)
a_c_z = Symbol("a_c_z", real=True)              # acc-z command (body-frame)
a_c = Matrix([a_c_x, a_c_y, a_c_z])

# specific force
# this is the physical quantity measured by
# an accelerometer.
accel_x = Symbol("accel_x", real=True)
accel_y = Symbol("accel_y", real=True)
accel_z = Symbol("accel_z", real=True)
accel = Matrix([accel_x, accel_y, accel_z])

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

C_eci_to_body = C_body_to_eci.T

##################################################

# (7) Earth-relative position vector
r_e_e = C_eci_to_ecef * r_i_m

# (9)
omega_skew_ie = Matrix([
    [0, -omega_e, 0],
    [omega_e, 0, 0],
    [0, 0, 0],
    ])

# (8) Earth-relative velocity vector
v_e_e = C_eci_to_ecef * v_i_m - omega_skew_ie * r_e_e

##################################################

# (12)
epsilon = 1e-3
V = v_e_e.norm()
# V = Piecewise(
#         (sp.Expr(V), is_launched),
#         (0.0, True)
#         )

# (10) Mach number
M = V / C_s
# zero-protect
M = Piecewise(
        (M, C_s > 0.0),  # type:ignore
        (0, True)
        )

# (11) Dynamic pressure
q_bar = 0.5 * rho * V**2  # type:ignore

##################################################
# invert aerodynamics
##################################################
C_ecef_to_body = C_body_to_ecef.T

f_b_T = Matrix([thrust, 0, 0])

Sref = aerotable.get_Sref()
CYB = aerotable.get_CYB(alpha=alpha, beta=beta, phi=np.nan, mach=M, iota=np.nan)
CNB = aerotable.get_CNB(alpha=alpha, beta=beta, phi=np.nan, mach=M, alt=alt, iota=np.nan)
# CA = aerotable.get_CA(alpha=alpha, phi=np.nan, mach=M, alt=alt, iota=np.nan, thrust=thrust)
CA = Piecewise(
        (aerotable.get_CA_Boost(alpha=alpha, beta=beta, phi=np.nan, mach=M, alt=alt, iota=np.nan), thrust > 0.),
        (aerotable.get_CA_Coast(alpha=alpha, beta=beta, phi=np.nan, mach=M, alt=alt, iota=np.nan), True)
        )

f_b_A_x = CA * q_bar * Sref
f_b_A_y = CYB * q_bar * Sref
f_b_A_z = CNB * q_bar * Sref
f_b_A = Matrix([f_b_A_x, f_b_A_y, f_b_A_z])

# (6)
# g_i_m = Matrix([gacc, 0, 0])
# g_e_m = C_eci_to_ecef * g_i_m
# g_b_e = C_ecef_to_body * g_e_m
r_hat = r_i_m / r_i_m.norm()
g_i_m = gacc * r_hat
g_b_e = C_eci_to_body * g_i_m

# a_b_m_expr = ((f_b_A + f_b_T) / wet_mass) + g_b_e
a_b_m_expr = ((f_b_T - f_b_A) / wet_mass) + g_b_e
a_b_m = Matrix([
    Piecewise(
        (0.0, sp.Eq(is_launched, 0)),
        (a_b_m_expr[0], True)),
    Piecewise(
        (0.0, sp.Eq(is_launched, 0)),
        (a_b_m_expr[1], True)),
    Piecewise(
        (0.0, sp.Eq(is_launched, 0)),
        (a_b_m_expr[2], True))
    ])

# (13) Earth-relative acceleration vector
a_e_e = (C_body_to_ecef * a_b_m - (2 * omega_skew_ie * v_e_e)
         - (omega_skew_ie * omega_skew_ie * r_e_e))

# (29)
a_b_e = C_ecef_to_body * a_e_e

# for specific force measurements (accelerometer)
accel_meas = ((f_b_T - f_b_A) / wet_mass) + g_b_e

##################################################

# (32) (33) (34)
u_hat = (1 + tan(alpha)**2 + tan(beta)**2)**(-0.5)  # type:ignore
v_hat = u_hat * tan(beta)
w_hat = u_hat * tan(alpha)

# (35) (36) (37)
u_hat_dot = (-(1 + tan(alpha)**2 + tan(beta)**2)**(-3 / 2)  # type:ignore # noqa
             * ((alpha_dot * tan(alpha) * sec(alpha)**2)
                + (beta_dot * tan(beta) * sec(beta)**2)))
v_hat_dot = u_hat_dot * tan(beta) + u_hat * beta_dot * sec(beta)**2
w_hat_dot = u_hat_dot * tan(alpha) + u_hat * alpha_dot * sec(alpha)**2

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

# zero-protect
q_new = Piecewise(
        (q_new, Abs(u) > 0.0),  # type:ignore
        (q, True)
        )
r_new = Piecewise(
        (r_new, Abs(u) > 0.0),  # type:ignore
        (r, True)
        )

##################################################

# (26)
omega_b_ib = Matrix([p, q, r])

##################################################

# (2)
r_i_m_dot = v_i_m

# (3)
v_i_m_dot = C_body_to_eci * a_b_e

##################################################
# 2.2 MMD Autopilot Transfer Functions
##################################################

# (16) Angle of attack transfer function
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

# (19) Pseudo-roll angle
phi_hat_dot = p

# NOTE: this is for BTT
# phi_new = atan2(C_eci_to_body[1, 2], C_eci_to_body[2, 2])

# (24)
# phi_dot = (1 / T_r) * (phi_c - phi_new)  # type:ignore

# (22)
phi_hat_c = phi_c - p

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

# (28)
# omega_skew_ib = Matrix([
#     [0, -r, q],
#     [r, 0, -q],
#     [-q, p, 0]
#     ])

# (27)
# TODO: include this or not?
# a_b_e = vbe_dot + (omega_skew_ib - C_b_to_e.T * omega_skew_ie * C_b_to_e) * vbe

# (46) NOTE: this is for a BTT missile

##################################################
# 3.1 Tail-Control Guidance Command Mapping
##################################################

# (48) Total life coefficient command
C_N_c = (a_c * wet_mass) / q_bar * Sref
# zero-protect
C_N_c = Piecewise(
        (C_N_c, q_bar > 0.0),
        (0, True)
        )

# (49) Aerodynamics roll angle command
phi_Ac = atan2(-a_c_y, -a_c_z)

# (50)
a_c_new = sqrt(a_c_y**2 + a_c_z**2)  # type:ignore

##################################################
# Aerodynamics Inversion
##################################################

alpha_total = aerotable.inv_aerodynamics(
        thrust=thrust,  # type:ignore
        acc_cmd=a_c_new,
        beta=beta,
        dynamic_pressure=q_bar,
        mass=wet_mass,
        alpha=alpha,
        phi=phi_hat,
        mach=M,
        alt=alt,
        iota=0.0,  # iota
        )

# (51)
alpha_c_new = atan(tan(alpha_total) * cos(phi_Ac))  # type:ignore
beta_c_new = atan(tan(alpha_total) * sin(phi_Ac))  # type:ignore

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
r_enu_e_new = C_ecef_to_enu * (r_e_e - ecef0)

# NOTE: non-position vectors should use the current vehicle
# position as the reference point
lla0 = ecef_to_lla_sym(r_e_e)
lat0, lon0, alt0 = lla0
C_ecef_to_enu = Matrix([
    [-sin(lon0), cos(lon0), 0],  # type:ignore
    [-sin(lat0) * cos(lon0), -sin(lat0) * sin(lon0), cos(lat0)],  # type:ignore
    [cos(lat0) * cos(lon0), cos(lat0) * sin(lon0), sin(lat0)]  # type:ignore
    ])
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
       (r_i_m.diff(t), r_i_m_dot),
       (v_i_m.diff(t), v_i_m_dot),

       (alpha_state.diff(t), alpha_state_dot),
       (beta_state.diff(t), beta_state_dot),
       (phi_hat.diff(t), phi_hat_dot),

       (p.diff(t), p_dot),

       (r_e_m.diff(t), v_e_e),
       (v_e_m.diff(t), a_e_e),

       (vel_mag_e.diff(t), V_dot),

       (v_b_e_m.diff(t), v_b_e_dot),
       (v_b_e_m_hat.diff(t), v_b_e_hat_dot),

       (wet_mass.diff(t), mass_dot),

       # (omega_n, 20),  # natural frequency
       # (zeta, 0.7),    # damping ratio
       # (K_phi, 1),     # roll gain
       # (omega_p, 20),  # natural frequency (roll)
       # (phi_c, 0),     # roll angle command
       # (T_r, 0.5),     # roll autopilot time constant

       (C_s, atmosphere.speed_of_sound(alt=alt)),
       (rho, atmosphere.density(alt=alt)),
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
    q_m,  # 0 - 3
    r_i_m,  # 4 - 6
    v_i_m,  # 7 - 9
    DirectUpdate(a_i_m, v_i_m_dot),

    alpha,      # 13
    alpha_dot,  # 14
    beta,       # 15
    beta_dot,   # 16
    phi_hat,    # 17
    DirectUpdate("phi_hat_dot", phi_hat_dot),  # 18

    # Angular rates
    p,  # 19
    DirectUpdate(q, q_new),  # 20
    DirectUpdate(r, r_new),  # 21

    # ENU
    DirectUpdate(r_enu_m, r_enu_e_new),         # 21 - 24
    DirectUpdate(v_enu_m, v_enu_e_new),         # 25 - 27
    DirectUpdate(a_enu_m, a_enu_e_new),         # 28 - 30

    # ECEF
    r_e_m,  # 31 -33
    v_e_m,  # 34 - 36
    DirectUpdate(a_e_m, a_e_e),  # 37 - 39

    vel_mag_e,  # 40
    DirectUpdate(vel_mag_e_dot, V_dot),  # 41
    DirectUpdate(mach, M),  # 42

    v_b_e_m,  # 43 - 45
    v_b_e_m_hat,  # 46 - 48

    DirectUpdate(g_b_m, g_b_e),  # 49 - 51
    DirectUpdate(a_b_e_m, a_b_m),  # 52 - 54

    # Mass Properties
    wet_mass,  # 55
    dry_mass,  # 56

    DirectUpdate("CA", CA),  # 57
    DirectUpdate("CN", CNB),  # 58
    DirectUpdate("q_bar", q_bar),  # 59

    DirectUpdate(drag, f_b_A[0]),  # 60 - 62
    DirectUpdate(slip, f_b_A[1]),  # 60 - 62
    DirectUpdate(lift, f_b_A[2]),  # 63 - 65
    DirectUpdate(accel, accel_meas),

    DirectUpdate("rho", rho),
    ])

input = Matrix([
    DirectUpdate(alpha_c, alpha_c_new),
    DirectUpdate(beta_c, beta_c_new),
    a_c_y,
    a_c_z,
    thrust,
    mass_dot,
    gacc,
    ])

static = Matrix([
    omega_n,  # natural frequency
    zeta,     # damping ratio
    K_phi,    # roll gain
    omega_p,  # natural frequency (roll)
    phi_c,    # roll angle command
    T_r,      # roll autopilot time constant
    is_boosting,
    stage,
    is_launched,
    ])
##################################################
# Define dynamics
##################################################

dynamics = state.diff(t)

##################################################
# Auto-Detect DirectUpdates
##################################################

# # quat = np.array([.7, 0, .7, 0])
# # n = np.linalg.norm(quat)
# # nquat = quat / n

# ####
# p, q, r = symbols("p, q, r")
# omega = Matrix([p, q, r])
# Sq = Matrix([[-q_1, -q_2, -q_3],    # type:ignore
#              [q_0, -q_3, q_2],      # type:ignore
#              [q_3, q_0, -q_1],      # type:ignore
#              [-q_2, q_1, q_0]])     # type:ignore

# subs = {dt: 0.1,
#         p: 1,
#         q: 0,
#         r: 0,
#         q_0: 1,
#         q_1: 0,
#         q_2: -1,
#         q_3: 0,
#         }
# q_m_dot = 0.5 * Sq * omega
# q_new = q_m + q_m_dot * dt
# nq_new = Quaternion(*q_new, norm=1).normalize()

# ssubs = {
#         q_0.diff(t): q_m_dot.doit()[0],
#         q_1.diff(t): q_m_dot.doit()[1],
#         q_2.diff(t): q_m_dot.doit()[2],
#         q_3.diff(t): q_m_dot.doit()[3],
#         }
# ####

# # def_vars = flatten_list([var for (var, _) in defs])
# # diff_defs = []
# # direct_defs = []
# # for var in def_vars:
# #     if var.is_Derivative:
# #         diff_defs += [var]
# #     else:
# #         direct_defs += [var]
# quit()

##################################################
# Build Model
##################################################

modules = [atmosphere.modules,
           aerotable.modules]

if __name__ == "__main__":
    model = MissileGenericMMD.from_expression(dt,
                                              state,
                                              input,
                                              dynamics,
                                              static_vars=static,
                                              modules=modules,
                                              definitions=defs,
                                              use_multiprocess_build=True)

    ##################################################
    # Python CodeGen
    ##################################################
    # # model.save(path=JAPL_HOME_DIR + "/../mmd/", name="mmd")

    path = "./"
    # imports = ["from config import aerotable_get_CA",
    #            "from config import aerotable_get_CA_Boost",
    #            "from config import aerotable_get_CA_Coast",
    #            "from config import aerotable_get_CNB",
    #            "from config import aerotable_get_CYB",
    #            "from config import aerotable_get_Sref",
    #            "from config import atmosphere_density",
    #            "from config import atmosphere_speed_of_sound",
    #            "from config import aerotable_inv_aerodynamics"]
    imports = ["from aero import aerotable",
               "from config import atmosphere"]

    to_pycode(func_name="dynamics_func",
              expr=model.dynamics_expr,
              state_vars=state,
              input_vars=input,
              static_vars=static,
              filepath=os.path.join(path, "mmd_dynamics.py"),
              imports=imports)

    to_pycode(func_name="state_update_func",
              expr=model.state_direct_updates,
              state_vars=state,
              input_vars=input,
              static_vars=static,
              filepath=os.path.join(path, "mmd_state_update.py"),
              imports=imports)

    to_pycode(func_name="input_update_func",
              expr=model.input_direct_updates,
              state_vars=state,
              input_vars=input,
              static_vars=static,
              filepath=os.path.join(path, "mmd_input_update.py"),
              imports=imports)

    ##################################################
    # C++ CodeGen
    ##################################################
    # model.create_c_module(name="mmd", path="./")
