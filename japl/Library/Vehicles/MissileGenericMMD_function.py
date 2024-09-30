import dill
import os
import numpy as np
from sympy import Matrix, Symbol, symbols
# from sympy import sign, rad, sqrt
# from sympy import sin, cos
# from sympy import atan, atan2, tan
# from sympy import sec
# from sympy import Float
# import sympy as sp
from japl import Model
from sympy import Function
from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.Aero.Atmosphere import Atmosphere
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl.Aero.AeroTableSymbolic import AeroTableSymbolic
from japl.Aero.AeroTable import AeroTable
from japl.Math.RotationSymbolic import ecef_to_lla_sym
from japl.Math import Rotation
from japl.Library.Earth.Earth import Earth
from japl.Util.Desym import Desym
from japl.Sim.Integrate import runge_kutta_4
from japl.Util.Matlab import MatFile
from scipy.interpolate import RegularGridInterpolator

DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=8)


class MissileGenericMMD(Model):
    pass


@DeprecationWarning
def calc_burn_ratio(mass, wet_mass, dry_mass):
    return (wet_mass - mass) / (wet_mass - dry_mass)


def create_mass_props(stage_table: MatFile):
    nozzle_area = stage_table.get("NozzleArea")
    mass_dot = stage_table.get("Mdot")
    cg = stage_table.get("CG")
    dry_mass = stage_table.get("DryMass")
    wet_mass = stage_table.get("WetMass")
    vac_flag = stage_table.get("VacFlag")
    thrust = stage_table.get("Thrust")
    prop_mass = stage_table.get("PropMass")
    burn_time = stage_table.get("StageTime")
    # calc burn ratio axis for interp
    # burn_ratio = (np.array([wet_mass] * len(prop_mass)) - prop_mass  # type:ignore
    #               - np.array([dry_mass] * len(prop_mass))) / (wet_mass - dry_mass)  # type:ignore
    mass_props = {
            "nozzle_area": nozzle_area,
            "mass_dot": RegularGridInterpolator((burn_time,), mass_dot),
            "cg": RegularGridInterpolator((burn_time,), cg),
            "dry_mass": dry_mass,
            "wet_mass": wet_mass,
            "vac_flag": vac_flag,
            "thrust": RegularGridInterpolator((burn_time,), thrust),
            "prop_mass": prop_mass,
            "burn_time": burn_time,
            }
    return mass_props


################################################
# Tables
################################################

atmosphere = Atmosphere()
# aerotable = AeroTable(DIR + "/../../../aeromodel/cms_sr_stage1aero.mat",
#                       from_template="CMS")
aerotable = AeroTable(DIR + "/../../../aeromodel/stage_1_aero.mat",
                      from_template="CMS")
stage_1_aero = MatFile(DIR + "/../../../aeromodel/stage_1_aero.mat")
stage_1_mass = MatFile(DIR + "/../../../aeromodel/stage_1_mass.mat")

mass_props = create_mass_props(stage_1_mass)

# ret = aerotable.ld_guidance(alpha=0.1, mach=1.0, alt=1000)

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

# Angle-of-Attack & Sideslip-Angle
alpha = Function("alpha", real=True)(t)
alpha_dot = Function("alpha_dot", real=True)(t)
alpha_dot_dot = Function("alpha_dot_dot", real=True)(t)

beta = Function("beta", real=True)(t)
beta_dot = Function("beta_dot", real=True)(t)
beta_dot_dot = Function("beta_dot_dot", real=True)(t)

alpha_c = Symbol("alpha_c", real=True)  # angle of attack command
beta_c = Symbol("beta_c", real=True)  # sideslip angle command

phi_hat, phi_hat_dot = symbols("phi_hat, phi_hat_dot", real=True)
a_c_x, a_c_y, a_c_z = symbols("a_c_x, a_c_y, a_c_z", real=True)

# angular velocities (roll, pitch, yaw)
p = Function("p", real=True)(t)
q = Function("q", real=True)(t)
r = Function("r", real=True)(t)

mass = Symbol("mass", real=True)
mach = Function("mach", real=True)(t)
C_s = Symbol("C_s", real=True)  # speed of sound
rho = Symbol("rho", real=True)  # speed of sound

omega_n = Symbol("omega_n", real=True)  # natural frequency
zeta = Symbol("zeta", real=True)  # damping ratio

K_phi = Symbol("K_phi", real=True)       # roll gain
omega_p = Symbol("omega_p", real=True)   # natural frequency (roll)
phi_c = Symbol("phi_c", real=True)       # roll angle command
T_r = Symbol("T_r", real=True)           # roll autopilot time constant

Sref = Symbol("Sref", real=True)

##################################################
# Vars
##################################################

state = [
        *q_m,
        *r_i_m,
        *v_i_m,
        alpha,
        alpha_dot,
        beta,
        beta_dot,
        p,
        q,
        r,
        mass,
        *r_enu_m,
        *v_enu_m,
        *a_enu_m,
        *r_e_m,
        *v_e_m,
        *a_e_m,
        vel_mag_e,
        vel_mag_e_dot,
        mach,
        v_b_e_m,
        v_b_e_m_hat,
        g_b_m,
        a_b_e_m,
        ]

input = [
        alpha_c,
        beta_c,
        a_c_y,
        a_c_z,
        thrust,
        ]

vars = (t, state, input, dt)

boosting = True
apogee = False


def sec(x):
    return (1.0 / np.cos(x))


def update_func(t, X: np.ndarray, U: np.ndarray, dt):
    global omega_e
    global atmosphere
    global boosting

    q_0, q_1, q_2, q_3 = X[:4]
    r_i_m = X[4:7]
    v_i_m = X[7:10]
    # alpha = X[10]
    # alpha_dot = X[11]
    # beta = X[12]
    # beta_dot = X[13]
    # p = X[14]
    # q = X[15]
    # r = X[16]
    mass = X[17]
    r_enu_m = X[18:21]
    # v_enu_m = X[21:24]
    # a_enu_m = X[24:27]
    r_e_m = X[27:30]
    v_e_m = X[30:33]
    a_e_m = X[33:36]
    # vel_mag_e = X[36]
    # vel_mag_e_dot = X[37]
    # mach = X[38]
    # v_b_e_m = X[39:42]
    # v_b_e_m_hat = X[42:45]
    # g_b_m = X[45:48]
    # a_b_e_m = X[48:51]

    alt = r_enu_m[2]

    # inputs
    # alpha_c = U[0]
    # beta_c = U[1]
    # a_c_y = U[2]
    # a_c_z = U[3]
    thrust = U[4]

    # other
    # omega_n = 50
    # zeta = 0.7
    # K_phi = 1
    # omega_p = 20
    # phi_c = 0
    # T_r = 0.5


    ##################################################
    # 2.1 ECI Position and Velocity Derivatives
    ##################################################

    # (1)
    C_eci_to_ecef = np.array([
        [np.cos(omega_e * t), np.sin(omega_e * t), 0],
        [-np.sin(omega_e * t), np.cos(omega_e * t), 0],
        [0, 0, 1]])

    # (4)
    C_body_to_eci = np.array([
        [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],  # type:ignore # noqa
        [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],  # type:ignore # noqa
        [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa

    # (14)
    C_body_to_ecef = C_eci_to_ecef @ C_body_to_eci

    ##################################################

    # (7) Earth-relative position vector
    r_e_e = C_eci_to_ecef @ r_i_m

    # (9)
    omega_skew_ie = np.array([
        [0, -omega_e, 0],
        [omega_e, 0, 0],
        [0, 0, 0],
        ])

    # (8) Earth-relative velocity vector
    v_e_e = C_eci_to_ecef @ v_i_m - omega_skew_ie @ r_e_e

    ##################################################
    # (12)
    epsilon = 1e-3
    V = np.linalg.norm(v_e_e)
    # (10) Mach number
    C_s = atmosphere.speed_of_sound(alt)
    M = V / C_s
    # (11) Dynamic pressure
    # rho = atmosphere.density(alt)  # kg * m^-3
    q_bar = 0.5 * rho * V**2  # type:ignore
    ##################################################
    # invert aerodynamics
    ##################################################
    C_ecef_to_body = C_body_to_ecef.T
    f_b_T = np.array([thrust, 0, 0])
    # TODO compute f_b_A here
    f_b_A = np.array([0, 0, 0])
    # (6)
    gacc = -9.81
    g_i_m = np.array([gacc, 0, 0])
    g_e_m = C_eci_to_ecef @ g_i_m
    g_b_e = C_ecef_to_body @ g_e_m
    a_b_m = ((f_b_A + f_b_T) / mass) + g_b_e
    # (13) Earth-relative acceleration vector
    a_e_e = (C_body_to_ecef @ a_b_m - (2 * omega_skew_ie @ v_e_e)
             - (omega_skew_ie @ omega_skew_ie @ r_e_e))
    # (29)
    a_b_e = C_ecef_to_body @ a_e_e

    ##################################################

    ##################################################
    # ECI to ENU convesion
    ##################################################

    # eci0 = np.array([Earth.radius_equatorial, 0, 0])
    ecef0 = np.array([Earth.radius_equatorial, 0, 0])
    # ecef0 = C_eci_to_ecef * eci0
    lla0 = Rotation.ecef_to_lla(ecef0)
    lat0, lon0, alt0 = lla0
    C_ecef_to_enu = np.array([
        [-np.sin(lon0), np.cos(lon0), 0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ])
    r_enu_e_new = C_ecef_to_enu @ (r_e_m - ecef0)

    # NOTE: non-position vectors should use the current vehicle
    # position as the reference point
    lla0 = Rotation.ecef_to_lla(r_e_m)
    lat0, lon0, alt0 = lla0
    C_ecef_to_enu = np.array([
        [-np.sin(lon0), np.cos(lon0), 0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ])

    v_enu_e_new = C_ecef_to_enu @ v_e_m
    a_enu_e_new = C_ecef_to_enu @ a_e_m

    X[:4] = X[:4] / np.linalg.norm(X[:4])
    # X[10] = alpha
    # X[11] = alpha_dot
    # X[12] = beta
    # X[13] = beta_dot
    # X[15] = q_new
    # X[16] = r_new
    # X[17] = mass
    X[18:21] = r_enu_e_new
    X[21:24] = v_enu_e_new
    X[24:27] = a_enu_e_new
    X[33:36] = a_e_e
    # X[37] = V_dot
    X[38] = M
    X[45:48] = g_b_e
    X[48:51] = a_b_e

    return X


def input_update_func(t, X, U, dt):
    global boosting
    global mass_props
    global apogee

    q_0, q_1, q_2, q_3 = X[:4]
    alpha = X[10]
    mass = X[17]
    r_enu_m = X[18:21]
    r_e_m = X[27:30]
    v_e_m = X[30:33]
    mach = X[38]
    a_b_e_m = X[48:51]
    v_enu_m = X[21:24]

    alt = r_enu_m[2]
    v_e_e = v_e_m
    phi = 0

    # inputs
    alpha_c = U[0]
    beta_c = U[1]
    a_c_y = U[2]
    a_c_z = U[3]
    thrust = U[4]

    # input cmds
    #########################################
    C_eci_to_ecef = np.array([
        [np.cos(omega_e * t), np.sin(omega_e * t), 0],
        [-np.sin(omega_e * t), np.cos(omega_e * t), 0],
        [0, 0, 1]])
    C_body_to_eci = np.array([
        [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],  # type:ignore # noqa
        [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],  # type:ignore # noqa
        [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa
    # ecef0 = np.array([Earth.radius_equatorial, 0, 0])
    ecef0 = r_e_m
    # ecef0 = C_eci_to_ecef * eci0
    lla0 = Rotation.ecef_to_lla(ecef0)
    lat0, lon0, alt0 = lla0
    C_ecef_to_enu = np.array([
        [-np.sin(lon0), np.cos(lon0), 0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ])
    C_body_to_ecef = C_eci_to_ecef @ C_body_to_eci

    V = np.linalg.norm(v_e_e)
    # (10) Mach number
    C_s = atmosphere.speed_of_sound(alt)
    M = V / C_s
    rho = atmosphere.density(alt)  # kg * m^-3
    q_bar = 0.5 * rho * V**2  # type:ignore

    # LD guidance
    Sref = aerotable.get_Sref()
    opt_CL, opt_CD, opt_alpha = aerotable.ld_guidance(alpha=alpha, mach=mach, alt=alt)
    f_l = opt_CL * q_bar * Sref
    f_d = opt_CD * q_bar * Sref
    a_l = f_l / mass
    a_d = f_d / mass

    if t < 1:  # cheap pog
        a_c_y = 0
        a_c_z = 1
    else:
        a_c_y = 0
        a_c_z = 0
    # print(opt_alpha, opt_CL, opt_CD)

    if not apogee and v_enu_m[2] <= 0:
        apogee = True
        print("apogee reached @ %.2f" % t)

    # f_b_A_z = CNB * q_bar * Sref
    # f_b_A = np.array([-f_b_A_x, 0, -f_b_A_z])

    # acc_cmd_ecef = C_ecef_to_enu.T @ np.array([0, 100, 100])

    # if t > 3:
    #     acc_cmd_ecef = C_ecef_to_enu.T @ np.array([0, -1, 0])
    # # if t > 5:
    # #     acc_cmd_ecef = C_ecef_to_enu.T @ np.array([0, 0, -40])
    # if t > 5:
    #     acc_cmd_ecef = C_ecef_to_enu.T @ np.array([0, 50, 20])
    # if t > 10:
    #     acc_cmd_ecef = C_ecef_to_enu.T @ np.array([0, -50, 0])
    # if t > 15:
    #     acc_cmd_ecef = C_ecef_to_enu.T @ np.array([0, 50, 0])
    # if t > 20 and t < 30:
    #     boosting = True

    # acc_cmd_body = C_body_to_ecef.T @ acc_cmd_ecef

    # acc_cmd_body = 60 * (acc_cmd_body / np.linalg.norm(acc_cmd_body))
    # a_c_y = acc_cmd_body[1]
    # a_c_z = acc_cmd_body[2]
    #########################################

    a_c_y = 0
    a_c_z = 0
    phi_Ac = np.arctan2(-a_c_y, -a_c_z)
    a_c_new = np.sqrt(a_c_y**2 + a_c_z**2)

    if apogee:
    #     phi_Ac = np.arctan2(0, a_l)
    #     a_c_new = np.sqrt(a_d**2 + a_l**2)
        phi_Ac = np.arctan2(0, 0)
        a_c_new = 0

    alpha_total = aerotable.inv_aerodynamics(
            thrust,  # type:ignore
            a_c_new,
            q_bar,  # type:ignore
            mass,
            alpha,
            phi,
            M,  # type:ignore
            alt,
            0.0,  # iota
            )

    alpha_c_new = np.arctan(np.tan(alpha_total) * np.cos(phi_Ac))
    beta_c_new = np.arctan(np.tan(alpha_total) * np.sin(phi_Ac))

    # alpha_c_new = np.radians(-3)
    # beta_c_new = np.radians(-1)

    # print(alpha_c_new, beta_c_new)
    # print(a_c_y, a_c_z, a_c_new)

    #######################################################
    # mass prop stuff
    #######################################################
    # mass
    # mass_dot
    # cg
    # thrust
    # vac_thrust
    wet_mass = mass_props["wet_mass"]
    dry_mass = mass_props["dry_mass"]
    vac_thrust, _thrust, mass_dot, isp = get_thrust(t, mass_props, atmosphere.pressure(alt))
    # print(_thrust)

    if boosting and (abs(mass) - abs(dry_mass)) < 0.001:  # type:ignore
        boosting = False
        print("boost end @ %.2f" % t)
    # if boosting and t > 1.9665:
    #     boosting = False
    #     print("boost end @ %.2f" % t)

    #######################################################

    if not boosting:
        _thrust = 0

    U[0] = alpha_c_new
    U[1] = beta_c_new
    U[2] = a_c_y
    U[3] = a_c_z
    U[4] = _thrust
    return U


def dynamics_func(t, X, U, dt):
    global omega_e
    global atmosphere
    global boosting

    q_0, q_1, q_2, q_3 = X[:4]
    r_i_m = X[4:7]
    v_i_m = X[7:10]
    alpha = X[10]
    alpha_dot = X[11]
    beta = X[12]
    beta_dot = X[13]
    p = X[14]
    q = X[15]
    r = X[16]
    mass = X[17]
    r_enu_m = X[18:21]
    v_enu_m = X[21:24]
    a_enu_m = X[24:27]
    r_e_m = X[27:30]
    v_e_m = X[30:33]
    a_e_m = X[33:36]
    vel_mag_e = X[36]
    vel_mag_e_dot = [37]
    mach = X[38]
    v_b_e_m = X[39:42]
    v_b_e_m_hat = X[42:45]
    g_b_m = X[45:48]
    a_b_e_m = X[48:51]

    alt = r_enu_m[2]

    # inputs
    alpha_c = U[0]
    beta_c = U[1]
    a_c_y = U[2]
    a_c_z = U[3]
    thrust = U[4]

    # other
    omega_n = 20
    zeta = .7
    K_phi = 1
    omega_p = 20
    phi_c = 0
    T_r = 0.5

    ###################################
    C_eci_to_ecef = np.array([
        [np.cos(omega_e * t), np.sin(omega_e * t), 0],
        [-np.sin(omega_e * t), np.cos(omega_e * t), 0],
        [0, 0, 1]])
    C_body_to_eci = np.array([
        [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],  # type:ignore # noqa
        [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],  # type:ignore # noqa
        [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa
    C_body_to_ecef = C_eci_to_ecef @ C_body_to_eci
    C_ecef_to_body = C_body_to_ecef.T

    r_e_e = C_eci_to_ecef @ r_i_m
    omega_skew_ie = np.array([
        [0, -omega_e, 0],
        [omega_e, 0, 0],
        [0, 0, 0],
        ])
    v_e_e = C_eci_to_ecef @ v_i_m - omega_skew_ie @ r_e_e

    V = np.linalg.norm(v_e_e)
    C_s = atmosphere.speed_of_sound(alt)
    M = V / C_s
    rho = atmosphere.density(alt)  # kg * m^-3
    q_bar = 0.5 * rho * V**2  # type:ignore
    ##################################################
    # invert aerodynamics
    ##################################################
    C_ecef_to_body = C_body_to_ecef.T

    f_b_T = np.array([thrust, 0, 0])

    # Compute f_b_A here
    Sref = aerotable.get_Sref()
    CNB = aerotable.get_CNB(alpha=alpha, phi=0, mach=mach)

    if boosting:
        CA = aerotable.get_CA_Boost(alpha=alpha, phi=0, mach=mach, alt=alt)
    else:
        CA = aerotable.get_CA_Coast(alpha=alpha, phi=0, mach=mach, alt=alt)

    f_b_A_x = CA * q_bar * Sref
    f_b_A_z = CNB * q_bar * Sref
    f_b_A = np.array([-f_b_A_x, 0, -f_b_A_z])

    # L/D guidance

    gacc = -9.81
    g_i_m = np.array([gacc, 0, 0])
    g_e_m = C_eci_to_ecef @ g_i_m
    g_b_e = C_ecef_to_body @ g_e_m
    a_b_m = ((f_b_A + f_b_T) / mass) + g_b_e
    a_e_e = (C_body_to_ecef @ a_b_m - (2 * omega_skew_ie @ v_e_e)
             - (omega_skew_ie @ omega_skew_ie @ r_e_e))
    a_b_e = C_ecef_to_body @ a_e_e

    u_hat = (1 + np.tan(alpha)**2 + np.tan(beta)**2)**(-0.5)  # type:ignore
    v_hat = u_hat * np.tan(beta)
    w_hat = u_hat * np.tan(alpha)

    u_hat_dot = -(1 + np.tan(alpha)**2 + np.tan(beta)**2)**(-3 / 2) * ((alpha_dot * np.tan(alpha) * sec(alpha)**2) + (beta_dot * np.tan(beta) * sec(beta)**2)) # type:ignore # noqa
    v_hat_dot = u_hat_dot * np.tan(beta) + u_hat * beta_dot * sec(beta)**2  # type:ignore
    w_hat_dot = u_hat_dot * np.tan(alpha) + u_hat * alpha_dot * sec(alpha)**2  # type:ignore

    v_b_e_hat = np.array([u_hat, v_hat, w_hat])

    v_b_e_hat_dot = np.array([u_hat_dot, v_hat_dot, w_hat_dot])

    v_b_e = V * v_b_e_hat
    u = v_b_e[0]
    v = v_b_e[1]
    w = v_b_e[2]

    V_dot = a_b_e @ v_b_e_hat

    v_b_e_dot = V_dot * v_b_e_hat + V * v_b_e_hat_dot
    # u_dot = v_b_e_dot[0]
    v_dot = v_b_e_dot[1]
    w_dot = v_b_e_dot[2]

    C_11 = C_body_to_ecef.T[0, 0]
    C_12 = C_body_to_ecef.T[0, 1]
    # C_13 = C_body_to_ecef.T[0, 2]
    C_21 = C_body_to_ecef.T[1, 0]
    C_22 = C_body_to_ecef.T[1, 1]
    # C_23 = C_body_to_ecef.T[1, 2]
    C_31 = C_body_to_ecef.T[2, 0]
    C_32 = C_body_to_ecef.T[2, 1]
    # C_33 = C_body_to_ecef.T[2, 2]
    C_1 = (C_11 * C_32 - C_12 * C_31) * u + (C_21 * C_32 - C_22 * C_31) * v
    C_2 = (C_11 * C_22 - C_12 * C_21) * u + (C_22 * C_31 - C_21 * C_32) * w

    # u = np.clip(u, 0.1, np.inf)

    q_new = (w_dot - a_b_e[2] + p * v - omega_e * C_1) / u
    r_new = (a_b_e[1] - v_dot + p * w + omega_e * C_2) / u

    omega_b_ib = np.array([p, q_new, r_new])
    Sq = np.array([[-q_1, -q_2, -q_3],    # type:ignore
                   [q_0, -q_3, q_2],      # type:ignore
                   [q_3, q_0, -q_1],      # type:ignore
                   [-q_2, q_1, q_0]])     # type:ignore
    q_m_dot = 0.5 * Sq @ omega_b_ib

    r_i_m_dot = v_i_m
    v_i_m_dot = C_body_to_eci @ a_b_e

    alpha_state = np.array([alpha, alpha_dot])
    A_alpha = np.array([
        [0, 1],
        [-omega_n**2, -2 * zeta * omega_n]  # type:ignore
        ])
    B_alpha = np.array([
        [0],
        [omega_n**2]
        ])
    alpha_state_dot = A_alpha @ alpha_state + (B_alpha * alpha_c).flatten()

    # (17) Sideslip angle transfer function
    beta_state = np.array([beta, beta_dot])
    A_beta = np.array([
        [0, 1],
        [-omega_n**2, -2 * zeta * omega_n]  # type:ignore
        ])
    B_beta = np.array([
        [0],
        [omega_n**2]
        ])
    beta_state_dot = A_beta @ beta_state + (B_beta * beta_c).flatten()

    C_eci_to_body = C_body_to_eci.T
    phi = np.arctan2(C_eci_to_body[1, 2], C_eci_to_body[2, 2])
    phi_hat = (1 / T_r) * (phi_c - phi)  # type:ignore
    phi_hat_c = phi_c - phi  # type:ignore
    p_c = K_phi * (phi_hat_c - phi_hat)
    p_dot = omega_p * (p_c - p)

    #######################################################
    wet_mass = mass_props["wet_mass"]
    dry_mass = mass_props["dry_mass"]
    vac_thrust, _thrust, mass_dot, isp = get_thrust(t, mass_props, atmosphere.pressure(alt))

    X[15] = q_new
    X[16] = r_new

    Xdot = np.zeros_like(X)

    Xdot[:4] = q_m_dot
    Xdot[4:7] = r_i_m_dot
    Xdot[7:10] = v_i_m_dot
    Xdot[10] = alpha_state_dot[0]
    Xdot[11] = alpha_state_dot[1]
    Xdot[12] = beta_state_dot[0]
    Xdot[13] = beta_state_dot[1]
    Xdot[14] = p_dot
    Xdot[17] = mass_dot
    Xdot[27:30] = v_e_e
    Xdot[30:33] = a_e_e
    Xdot[36] = V_dot
    Xdot[39:42] = v_b_e_dot
    Xdot[42:45] = v_b_e_hat_dot
    return Xdot

##################################################
# Definitions
##################################################

# defs = (
#        (q_m.diff(t), q_m_dot_reg),
#        (r_i_m.diff(t), v_i_m),
#        (v_i_m.diff(t), v_i_m_dot),

#        (p.diff(t), p_dot),
#        # (q.diff(t), q_new.diff(t)),
#        # (r.diff(t), r_new.diff(t)),

#        (r_e_m.diff(t), v_e_e),
#        (v_e_m.diff(t), a_e_e),

#        (alpha_state.diff(t), alpha_state_dot),
#        (beta_state.diff(t), beta_state_dot),

#        (omega_n, 50),
#        (zeta, 0.7),
#        (K_phi, 1),
#        (omega_p, 20),
#        (phi_c, 0),
#        (T_r, 0.5),
#        (C_s, atmosphere.speed_of_sound(alt)),
#        (rho, atmosphere.density(alt)),

#        (v_b_e_m.diff(t), v_b_e_dot),
#        (v_b_e_m_hat.diff(t), v_b_e_hat_dot),

#        (vel_mag_e.diff(t), V_dot),
#        )

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

# state = Matrix([
#     q_m,
#     r_i_m,
#     v_i_m,
#     alpha_state,
#     beta_state,
#     p,
#     DirectUpdate(q, q_new),
#     DirectUpdate(r, r_new),
#     mass,

#     # ENU
#     DirectUpdate(r_enu_m, r_enu_e_new),
#     DirectUpdate(v_enu_m, v_enu_e_new),
#     DirectUpdate(a_enu_m, a_enu_e_new),

#     # ECEF
#     # DirectUpdate(r_e_m, r_e_e),
#     # DirectUpdate(v_e_m, v_e_e),
#     # DirectUpdate(a_e_m, a_e_e),
#     r_e_m,
#     v_e_m,
#     DirectUpdate(a_e_m, a_e_e),

#     # DirectUpdate(vel_mag_e, V),
#     vel_mag_e,
#     DirectUpdate(vel_mag_e_dot, V_dot),

#     DirectUpdate(mach, M),

#     # DirectUpdate(v_b_e_m, v_b_e_hat),
#     v_b_e_m,
#     v_b_e_m_hat,

#     DirectUpdate(g_b_m, g_b_e),
#     DirectUpdate(a_b_e_m, a_b_e)
#     # v_b_e_m,
#     ])

# input = Matrix([
#     DirectUpdate(alpha_c, alpha_c_new),
#     DirectUpdate(beta_c, beta_c_new),
#     a_c_y,
#     a_c_z,
#     thrust,
#     ])

##################################################
# Define dynamics
##################################################

# state.subs(b4subs)

# dynamics = state.diff(t)

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

# modules = [atmosphere.modules,
#            aerotable.modules]

# model = MissileGenericMMD.from_expression(dt,
#                                           state,
#                                           input,
#                                           dynamics,
#                                           modules=modules,
#                                           definitions=defs,
#                                           use_multiprocess_build=True)


model = MissileGenericMMD.from_function(dt,
                                        Matrix(state),
                                        Matrix(input),
                                        dynamics_func=dynamics_func,
                                        state_update_func=update_func,
                                        input_update_func=input_update_func,
                                        )


# print("saving model...")
# with open(f"{DIR}/mmd2.pickle", 'wb') as f:
#     dill.dump(model, f)
