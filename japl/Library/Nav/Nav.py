import dill as pickle
import numpy as np
import sympy as sp
# from itertools import permutations
from sympy import Matrix, Symbol, symbols, cse
from sympy import MatrixSymbol
from sympy import Function
from sympy import Piecewise
from sympy import Eq, Gt, Ge, Lt, Le, Mod
from sympy import sqrt
from sympy import simplify
# from sympy import default_sort_key, topological_sort
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl import JAPL_HOME_DIR
from japl.Util.Util import profile
from japl import Model
from japl.BuildTools.CCodeGenerator import CCodeGenerator


################################################################
# Helper Methods
################################################################


def get_mat_upper(mat):
    # ret = []
    # n = mat.shape[0]
    # for i in range(n):
    #     for j in range(n):
    #         if i > j:
    #             pass
    #         else:
    #             ret += [mat[j, i]]
    # return np.array(ret)
    ncols = mat.shape[1]
    ids = np.triu_indices(ncols)
    return np.asarray(mat)[ids]


# def zero_mat_lower(mat):
#     ret = mat.copy()
#     for index in range(mat.shape[0]):
#         for j in range(mat.shape[0]):
#             if index > j:
#                 ret[index, j] = 0
#     return ret


# def mat_print(mat):
#     if isinstance(mat, Matrix):
#         mat = np.array(mat)
#     for row in mat:
#         print('[', end="")
#         for item in row:
#             if item == 0:
#                 print("%s, " % (' ' * 8), end="")
#             else:
#                 print("%.6f, " % item, end="")
#         print(']')
#     print()


# def mat_print_sparse(mat):
#     for i in range(mat.shape[0]):
#         print('[', end="")
#         for j in range(mat.shape[1]):
#             item = mat[i, j]
#             try:
#                 item = float(item)
#                 if item == 0:
#                     print("%s, " % " ", end="")
#                 else:
#                     print("%d, " % 1, end="")
#             except:  # noqa
#                 if isinstance(item, sp.Expr):
#                     print("%d, " % 1, end="")
#         print(']')
#     print()


# def array_print(mat):
#     print('[', end="")
#     for item in mat:
#         if item == 0:
#             print("%s, " % (' ' * 8), end="")
#         else:
#             print("%.6f, " % item, end="")
#     print(']')


# def update_subs(subs, arr):
#     for i, (k, v) in enumerate(subs.items()):
#         subs[k] = arr[i]


# def sort_recursive_subs(replace):
#     """
#     For recursive substitutions, the order of variable subs
#     must be sorted.

#     Arguments:
#         - replace: the first return of sympy.cse (list[tuple])

#     Returns:
#         - replace_subs: dict of substitutions
#     """
#     edges = [(i, j) for i, j in permutations(replace, 2) if i[1].has(j[0])]
#     replace_subs = topological_sort([replace, edges], default_sort_key)
#     return replace_subs


# def cse_subs(cse, state_subs, input_subs, cov_subs, var_subs):
#     replace, expr = cse
#     expr = expr[0]
#     replace_subs = sort_recursive_subs(replace)
#     for (var, sub) in tqdm(replace_subs):
#         expr = expr.subs(var, sub)
#     expr = expr.subs(state_subs).subs(input_subs).subs(cov_subs).subs(var_subs).subs(dt, dt_)
#     return expr


################################################################


def quat_to_dcm(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    Rot = Matrix([[1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3)    , 2*(q1*q3 + q0*q2)    ],    # noqa
                 [2*(q1*q2 + q0*q3)     , 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)    ],    # noqa
                 [2*(q1*q3-q0*q2)       , 2*(q2*q3 + q0*q1)    , 1 - 2*(q1**2 + q2**2)]])   # noqa
    return Rot


# def quat_mult(p, q):
#     r = Matrix([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
#                 p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
#                 p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
#                 p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]])
#     return r


# def create_cov_matrix(i, j):
#     if j >= i:
#         # return Symbol("P(" + str(i) + "," + str(j) + ")", real=True)
#         # legacy array format
#         return Symbol("P[" + str(i) + "][" + str(j) + "]", real=True)
#     else:
#         return 0


# def create_symmetric_cov_matrix(n):
#     # define a symbolic covariance matrix
#     P = Matrix(n, n, create_cov_matrix)
#     for index in range(n):
#         for j in range(n):
#             if index > j:
#                 P[index, j] = P[j, index]
#     return P


# def generate_kalman_gain_equations(P, state, observation, variance, varname: str = "K"):
#     H = Matrix([observation]).jacobian(state)
#     innov_var = H * P * H.T + Matrix([variance])
#     assert (innov_var.shape[0] == 1)
#     assert (innov_var.shape[1] == 1)
#     K = (P * H.T) / innov_var[0, 0]
#     K_simple = cse(K, symbols(f"{varname}0:1000"), optimizations="basic")
#     return K_simple


################################################################
################################################################
################################################################


t = Symbol('t', real=True)
dt = Symbol('dt', real=True)

##################################################
# States
##################################################

q0 = Symbol("q0", real=True)  # (t)
q1 = Symbol("q1", real=True)  # (t)
q2 = Symbol("q2", real=True)  # (t)
q3 = Symbol("q3", real=True)  # (t)
pos_n = Symbol("pos_n", real=True)  # (t)
pos_e = Symbol("pos_e", real=True)  # (t)
pos_d = Symbol("pos_d", real=True)  # (t)
vel_n = Symbol("vel_n", real=True)  # (t)
vel_e = Symbol("vel_e", real=True)  # (t)
vel_d = Symbol("vel_d", real=True)  # (t)

acc_bias_x, acc_bias_y, acc_bias_z = symbols("acc_bias_x, acc_bias_y, acc_bias_z", real=True)
angvel_bias_x, angvel_bias_y, angvel_bias_z = symbols("angvel_bias_x, angvel_bias_y, angvel_bias_z", real=True)

quat = Matrix([q0, q1, q2, q3])
pos = Matrix([pos_n, pos_e, pos_d])
vel = Matrix([vel_n, vel_e, vel_d])
acc_bias = Matrix([acc_bias_x, acc_bias_y, acc_bias_z])
angvel_bias = Matrix([angvel_bias_x, angvel_bias_y, angvel_bias_z])

state = Matrix([quat, pos, vel, acc_bias, angvel_bias])

##################################################
# Measurements
##################################################

z_gyro_x, z_gyro_y, z_gyro_z = symbols("z_gyro_x, z_gyro_y, z_gyro_z", real=True)
z_accel_x, z_accel_y, z_accel_z = symbols("z_accel_x, z_accel_y, z_accel_z", real=True)
z_gps_pos_x, z_gps_pos_y, z_gps_pos_z = symbols("z_gps_pos_x, z_gps_pos_y, z_gps_pos_z", real=True)
z_gps_vel_x, z_gps_vel_y, z_gps_vel_z = symbols("z_gps_vel_x, z_gps_vel_y, z_gps_vel_z", real=True)

z_gyro = Matrix([z_gyro_x, z_gyro_y, z_gyro_z])
z_accel = Matrix([z_accel_x, z_accel_y, z_accel_z])
z_gps_pos = Matrix([z_gps_pos_x, z_gps_pos_y, z_gps_pos_z])
z_gps_vel = Matrix([z_gps_vel_x, z_gps_vel_y, z_gps_vel_z])
z_gps = Matrix([z_gps_pos_x, z_gps_pos_y, z_gps_pos_z,
                z_gps_vel_x, z_gps_vel_y, z_gps_vel_z])

angvel_true = z_gyro - angvel_bias
acc_true = z_accel - acc_bias

dcm_to_earth = quat_to_dcm(quat)
dcm_to_body = dcm_to_earth.T
gravity_ef = Matrix([0, 0, -9.81])  # gravity earth-frame
gravity_bf = dcm_to_body * gravity_ef

##################################################
# State Update Equations
##################################################
wx, wy, wz = angvel_true
Sw = Matrix([
    [0, wx, wy, wz],    # type:ignore
    [-wx, 0, -wz, wy],  # type:ignore
    [-wy, wz, 0, -wx],  # type:ignore
    [-wz, -wy, wx, 0],  # type:ignore
    ])

quat_new = quat + (-0.5 * Sw * quat) * dt
acc_world_measured = (dcm_to_earth * (acc_true - gravity_bf))
pos_new = pos + vel * dt + 0.5 * acc_world_measured * dt**2
vel_new = vel + acc_world_measured * dt
gyro_bias_new = angvel_bias
accel_bias_new = acc_bias

# NOTE: since no magnetometer yet, zero-out gyro_z bias
# gyro_bias_new[2] = sp.Float(0)

##################################################
# Process Noise
##################################################

gyro_x_var, gyro_y_var, gyro_z_var = symbols('gyro_x_var gyro_y_var gyro_z_var')
accel_x_var, accel_y_var, accel_z_var = symbols('accel_x_var accel_y_var accel_z_var')
gps_pos_x_var, gps_pos_y_var, gps_pos_z_var = symbols('gps_pos_x_var, gps_pos_y_var, gps_pos_z_var')
gps_vel_x_var, gps_vel_y_var, gps_vel_z_var = symbols('gps_vel_x_var, gps_vel_y_var, gps_vel_z_var')

# input variance
input_var = Matrix.diag([gyro_x_var, gyro_y_var, gyro_z_var,
                         accel_x_var, accel_y_var, accel_z_var,
                         gps_pos_x_var, gps_pos_y_var, gps_pos_z_var,
                         gps_vel_x_var, gps_vel_y_var, gps_vel_z_var])

##################################################
# State Prediction
##################################################

state_new = Matrix([quat_new, pos_new, vel_new, accel_bias_new, gyro_bias_new])
input = Matrix([z_gyro, z_accel, z_gps_pos, z_gps_vel])
F = state_new.jacobian(state)
G = state_new.jacobian(input)

X_new = F * state + G * input

Q = G * input_var * G.T

##################################################
# Covariance Prediction
##################################################

P = MatrixSymbol("P", len(state), len(state)).as_mutable()
# make P symmetric
for index in range(P.shape[0]):
    for j in range(P.shape[0]):
        if index > j:
            P[index, j] = P[j, index]

P_new = F * P * F.T + Q

# for index in range(P.shape[0]):
#     for j in range(P.shape[0]):
#         if index > j:
#             P_new[index, j] = 0

##################################################
# Observations
##################################################
print("Building Observations...")

# Observation Noise
R_accel_x, R_accel_y, R_accel_z = symbols('R_accel_x R_accel_y R_accel_z')
R_mag_world_x, R_mag_world_y, R_mag_world_z = symbols('R_mag_world_x R_mag_world_y R_mag_world_z')
R_gps_pos_x, R_gps_pos_y, R_gps_pos_z = symbols('R_gps_pos_x R_gps_pos_y R_gps_pos_z')
R_gps_vel_x, R_gps_vel_y, R_gps_vel_z = symbols('R_gps_vel_x R_gps_vel_y R_gps_vel_z')

# Body Frame Accelerometer Observation
# obs_accel = (dcm_to_earth * gravity_bf) - acc_bias
obs_accel = (dcm_to_body * gravity_ef) + acc_bias
# obs_accel = gravity_bf + acc_bias
H_accel_x = obs_accel[0, :].jacobian(state)
H_accel_y = obs_accel[1, :].jacobian(state)
H_accel_z = obs_accel[2, :].jacobian(state)
H_accel = Matrix([H_accel_x, H_accel_y, H_accel_z])

# Gps-position Observation (NED frame)
obs_gps_pos = pos_new
H_gps_pos_x = obs_gps_pos[0, :].jacobian(state)  # type:ignore
H_gps_pos_y = obs_gps_pos[1, :].jacobian(state)  # type:ignore
H_gps_pos_z = obs_gps_pos[2, :].jacobian(state)  # type:ignore
# H_gps_pos = Matrix([H_gps_pos_x, H_gps_pos_y, H_gps_pos_z])

# Gps-velocity Observation (NED frame)
obs_gps_vel = vel_new
H_gps_vel_x = obs_gps_vel[0, :].jacobian(state)  # type:ignore
H_gps_vel_y = obs_gps_vel[1, :].jacobian(state)  # type:ignore
H_gps_vel_z = obs_gps_vel[2, :].jacobian(state)  # type:ignore
# H_gps_vel = Matrix([H_gps_vel_x, H_gps_vel_y, H_gps_vel_z])

H_gps = Matrix([H_gps_pos_x, H_gps_pos_y, H_gps_pos_z,
                H_gps_vel_x, H_gps_vel_y, H_gps_vel_z])

##################################################
# Kalman Gains
##################################################
print("Building Kalman Gains...")

# Accelerometer
innov_var_accel_x = H_accel_x * P * H_accel_x.T + Matrix([R_accel_x])
innov_var_accel_y = H_accel_y * P * H_accel_y.T + Matrix([R_accel_y])
innov_var_accel_z = H_accel_z * P * H_accel_z.T + Matrix([R_accel_z])
K_accel_x = P * H_accel_x.T / innov_var_accel_x[0, 0]
K_accel_y = P * H_accel_y.T / innov_var_accel_y[0, 0]
K_accel_z = P * H_accel_z.T / innov_var_accel_z[0, 0]

# Gps-position
innov_var_gps_pos_x = H_gps_pos_x * P * H_gps_pos_x.T + Matrix([R_gps_pos_x])
innov_var_gps_pos_y = H_gps_pos_y * P * H_gps_pos_y.T + Matrix([R_gps_pos_y])
innov_var_gps_pos_z = H_gps_pos_z * P * H_gps_pos_z.T + Matrix([R_gps_pos_z])
K_gps_pos_x = P * H_gps_pos_x.T / innov_var_gps_pos_x[0, 0]
K_gps_pos_y = P * H_gps_pos_y.T / innov_var_gps_pos_y[0, 0]
K_gps_pos_z = P * H_gps_pos_z.T / innov_var_gps_pos_z[0, 0]

# Gps-velocity
innov_var_gps_vel_x = H_gps_vel_x * P * H_gps_vel_x.T + Matrix([R_gps_vel_x])
innov_var_gps_vel_y = H_gps_vel_y * P * H_gps_vel_y.T + Matrix([R_gps_vel_y])
innov_var_gps_vel_z = H_gps_vel_z * P * H_gps_vel_z.T + Matrix([R_gps_vel_z])
K_gps_vel_x = P * H_gps_vel_x.T / innov_var_gps_vel_x[0, 0]
K_gps_vel_y = P * H_gps_vel_y.T / innov_var_gps_vel_y[0, 0]
K_gps_vel_z = P * H_gps_vel_z.T / innov_var_gps_vel_z[0, 0]

##################################################
# Updates
##################################################
print("Building Updates...")

# kalman gain
K_accel = Matrix()
K_accel = K_accel.col_insert(0, K_accel_x)
K_accel = K_accel.col_insert(1, K_accel_y)
K_accel = K_accel.col_insert(2, K_accel_z)

K_gps = Matrix()
K_gps = K_gps.col_insert(0, K_gps_pos_x)
K_gps = K_gps.col_insert(1, K_gps_pos_y)
K_gps = K_gps.col_insert(2, K_gps_pos_z)
K_gps = K_gps.col_insert(3, K_gps_vel_x)
K_gps = K_gps.col_insert(4, K_gps_vel_y)
K_gps = K_gps.col_insert(5, K_gps_vel_z)

# accelerometer
X_accel_update = state + K_accel * (z_accel - H_accel * state)
P_accel_update = P - (K_accel * H_accel * P)

# gps
X_gps_update = state + K_gps * (z_gps - H_gps * state)
P_gps_update = P - (K_gps * H_gps * P)

state_info = {
        q0: 1,
        q1: 0,
        q2: 0,
        q3: 0,
        pos_n: 0,
        pos_e: 0,
        pos_d: 0,
        vel_n: 0,
        vel_e: 0,
        vel_d: 0,
        acc_bias_x: 0,
        acc_bias_y: 0,
        acc_bias_z: 0,
        angvel_bias_x: 0,
        angvel_bias_y: 0,
        angvel_bias_z: 0,
        }

# input_info = {
#         z_gyro[0, 0]: 0.0,        # type:ignore
#         z_gyro[1, 0]: 0.0,        # type:ignore
#         z_gyro[2, 0]: 0.0,        # type:ignore
#         z_accel[0, 0]: 0.0,        # type:ignore
#         z_accel[1, 0]: 0.0,        # type:ignore
#         z_accel[2, 0]: 0.0,        # type:ignore
#         }

# process noise
accel_Hz = 100
accel_bandwidth = accel_Hz / 2  # Nyquist frequency
accel_full_range = 100  # +-100 m/s^2
accel_noise_density = 10e-6  # 10 ug / sqrt(Hz)
# accel_var = (accel_noise_density * 9.81)**2 * accel_Hz
accel_var = accel_noise_density * np.sqrt(accel_bandwidth)

gyro_Hz = 100
gyro_bandwidth = gyro_Hz / 2  # Nyquist frequency
gyro_noise_density = 3.5e-3  # ((udeg / s) / sqrt(Hz))
gyro_var = gyro_noise_density * np.sqrt(gyro_bandwidth)

gps_pos_var = .1**2
gps_vel_var = .01**2

variance_info = {
        gyro_x_var: 10,
        gyro_y_var: 10,
        gyro_z_var: 10,
        accel_x_var: 10,
        accel_y_var: 10,
        accel_z_var: 10,
        gps_pos_x_var: .001,
        gps_pos_y_var: .001,
        gps_pos_z_var: .001,
        gps_vel_x_var: .001,
        gps_vel_y_var: .001,
        gps_vel_z_var: .001,
        }

# measurement noise
noise_info = {
        R_accel_x: accel_var,
        R_accel_y: accel_var,
        R_accel_z: accel_var,
        R_gps_pos_x: gps_pos_var,
        R_gps_pos_y: gps_pos_var,
        R_gps_pos_z: gps_pos_var,
        R_gps_vel_x: gps_vel_var,
        R_gps_vel_y: gps_vel_var,
        R_gps_vel_z: gps_vel_var,
        }

# sensor measurements
input_info = {
        z_gyro_x: 0,
        z_gyro_y: 0,
        z_gyro_z: 0,
        z_accel_x: 0,
        z_accel_y: 0,
        z_accel_z: 0,
        z_gps_pos_x: 0,
        z_gps_pos_y: 0,
        z_gps_pos_z: 0,
        z_gps_vel_x: 0,
        z_gps_vel_y: 0,
        z_gps_vel_z: 0,
        }
# TODO:
# TODO:
# TODO:
# TODO: input_subs accel and meas_subs z_accel both
#       existing is a problem because they are the same
#       thing. need to fix this.

#################################################
# Sympy lambdify funcs
#################################################

vars = [
        list(state_info.keys()),
        list(input_info.keys()),
        list(get_mat_upper(P)),
        list(variance_info.keys()),
        list(noise_info.keys()),
        # list(meas_info.keys()),
        dt,
        ]

if __name__ == "__main__":

    ##################################################
    # Write to File
    ##################################################

    # print('Simplifying covariance propagation ...')

    # args = symbols("P,"                         # covariance matrix
    #                "q0, q1, q2, q3,"            # quaternion
    #                "vn, ve, vd,"                # velocity in NED local frame
    #                "pn, pe, pd,"                # position in NED local frame
    #                "dvx, dvy, dvz,"             # delta velocity (accelerometer measurements)
    #                "dax, day, daz,"             # delta angle (gyroscope measurements)
    #                "dax_b, day_b, daz_b,"       # delta angle bias
    #                "dvx_b, dvy_b, dvz_b,"       # delta velocity bias
    #                "daxVar, dayVar, dazVar,"    # gyro input noise
    #                "dvxVar, dvyVar, dvzVar,"    # accel input noise
    #                "dt")

    # codegen = OctaveCodeGenerator()

    # print('Writing state propagation to file ...')
    # codegen.write_function_to_file(path=f"{JAPL_HOME_DIR}/derivation/nav/generated/state_predict.m",
    #                                function_name="state_predict",
    #                                expr=X_new,
    #                                input_vars=args,
    #                                return_var="nextX")


    # print('Writing covariance propagation to file ...')
    # codegen.write_function_to_file(path=f"{JAPL_HOME_DIR}/derivation/nav/generated/cov_predict.m",
    #                                function_name="cov_predict",
    #                                expr=P_new,
    #                                input_vars=args,
    #                                return_var="nextP")

    ##################################################
    # Sim
    ##################################################

    ######################################################
    # NOTE: trying to create model from_expression
    ######################################################
    # state_update = Matrix([
    #     DirectUpdate(state, X_new),
    #     DirectUpdate(P, P_new),
    #     ])

    static = Matrix([
        gyro_x_var,
        gyro_y_var,
        gyro_z_var,
        accel_x_var,
        accel_y_var,
        accel_z_var,
        gps_pos_x_var,
        gps_pos_y_var,
        gps_pos_z_var,
        gps_vel_x_var,
        gps_vel_y_var,
        gps_vel_z_var,
        ])

    X_new[0] = X_new[0] / quat_new.norm()
    X_new[1] = X_new[1] / quat_new.norm()
    X_new[2] = X_new[2] / quat_new.norm()
    X_new[3] = X_new[3] / quat_new.norm()

    # # X_new
    # # P_new
    # # X_accel_update
    # # P_accel_update
    # # X_gps_update
    # # P_gps_update

    noise = list(noise_info.keys())
    variance = list(variance_info.keys())
    input = list(input_info.keys())

    # upper_ids = np.triu_indices(P.shape[0])
    # upper_ncols = len(upper_ids[0])
    # P_upper = MatrixSymbol("P", upper_ncols, 1).as_mutable()
    # P_new_upper = Matrix([P_new[i, j] for i, j in zip(*upper_ids)])

    params = [t, state, input, P, variance, noise, dt]

    gen = CCodeGenerator()
    # gen.add_function(expr=X_new,
    #                  params=params,
    #                  function_name="x_predict",
    #                  return_name="X_new")

    # gen.add_function(expr=P_new,
    #                  params=params,
    #                  function_name="p_predict",
    #                  return_name="P_new",
    #                  is_symmetric=False)

    # gen.add_function(expr=X_accel_update,
    #                  params=params,
    #                  function_name="x_accel_update",
    #                  return_name="X_accel_new")

    # gen.add_function(expr=P_accel_update,
    #                  params=params,
    #                  function_name="p_accel_update",
    #                  return_name="P_accel_new",
    #                  is_symmetric=False)

    # gen.add_function(expr=X_gps_update,
    #                  params=params,
    #                  function_name="x_gps_update",
    #                  return_name="X_gps_new")

    gen.add_function(expr=P_gps_update,
                     params=params,
                     function_name="p_gps_update",
                     return_name="P_gps_new",
                     is_symmetric=False)

    gen.create_module(module_name="ekf", path="./")
    quit()

    definitions = (
            (state, X_new),
            (get_mat_upper(P), get_mat_upper(P_new)),
            )

    ekf_state = Matrix([
        state,
        *get_mat_upper(P),
        ])

    model = Model.from_expression(dt_var=dt,
                                  state_vars=ekf_state,
                                  input_vars=input,
                                  static_vars=static,
                                  definitions=definitions,
                                  dynamics_expr=Matrix([np.nan] * len(ekf_state)),
                                  use_multiprocess_build=True)
    model.save(f"{JAPL_HOME_DIR}/data", "ekf_expr")

    # t = 0
    # X = np.array([*state_info.values()])
    # U = np.array([*input_info.values()])
    # S = []
    # dt = 0.1
    # ret = model.direct_state_update_func(t, X, U, S, dt)
    # print(ret)

    # model = Model.from_file("./data/ekf.japl")
    # # pp = np.eye(P.shape[0])
    # # ret = model.direct_state_update_func(0, [1,0,0,0, 0,0,0, 0,1,0, 0,0,0, 0,0,0, pp],
    # #                                          [0,0,0, 0,0,0], [1,1,1, 1,1,1], 0.1)
    # # print(ret)

    # simobj = SimObject(model)
    # quat0 = [1, 0, 0, 0]
    # p0 = [0, 0, 0]
    # v0 = [0, 0, 0]
    # ab0 = [0, 0, 0]
    # gb0 = [0, 0, 0]
    # P0 = np.eye(P.shape[0])
    # simobj.init_state([quat0, p0, v0, ab0, gb0, P0])

    # sim = Sim(t_span=[0, 10],
    #           dt=0.01,
    #           simobjs=[simobj])
    # sim.run()

    # Y = simobj.Y[-1]
    # print(Y)
    # quit()
    ######################################################

    quit()

    print("Lambdify-ing Symbolic Expressions...")

    # state predict
    X_new_func = profile(sp.lambdify)(vars, X_new, cse=True)

    # covariance predict
    P_new_func = profile(sp.lambdify)(vars, P_new, cse=True)

    # update from accel
    X_accel_update_func = profile(sp.lambdify)(vars, X_accel_update, cse=True)
    P_accel_update_func = profile(sp.lambdify)(vars, P_accel_update, cse=True)

    # update from gps-position
    X_gps_update_func = profile(sp.lambdify)(vars, X_gps_update, cse=True)
    P_gps_update_func = profile(sp.lambdify)(vars, P_gps_update, cse=True)

    out = [("X_new_func", X_new_func),
           ("P_new_func", P_new_func),
           ("X_accel_update_func", X_accel_update_func),
           ("P_accel_update_func", P_accel_update_func),
           ("X_gps_update_func", X_gps_update_func),
           ("P_gps_update_func", P_gps_update_func)]

    for (name, func) in out:
        print(f"saving {name}...")
        with open(f"{JAPL_HOME_DIR}/data/{name}.pickle", "wb") as f:
            pickle.dump(func, f)
