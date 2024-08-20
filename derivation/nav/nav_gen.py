from tqdm import tqdm
import numpy as np
from sympy import Matrix, Symbol, symbols, cse, MatrixSymbol
import sympy as sp
# from code_gen import OctaveCodeGenerator
# from code_gen import CCodeGenerator
from code_gen import PyCodeGenerator
# import numpy as np
from pprint import pprint
from itertools import permutations
from sympy import default_sort_key, topological_sort



################################################################
# Helper Methods
################################################################

def get_mat_upper(mat):
    ret = []
    n = mat.shape[0]
    for i in range(n):
        for j in range(n):
            if i > j:
                pass
            else:
                ret += [mat[j, i]]
    return np.array(ret)


def mat_print(mat):
    for row in mat:
        print('[', end="")
        for item in row:
            if item == 0:
              print("%s, " % (' ' * 8), end="")
            else:
                print("%.6f, " % item, end="")
        print(']')
    print()


def update_subs(subs, arr):
    for i, (k, v) in enumerate(subs.items()):
        subs[k] = arr[i]


def sort_recursive_subs(replace):
    """
    For recursive substitutions, the order of variable subs
    must be sorted.

    Arguments:
        - replace: the first return of sympy.cse (list[tuple])

    Returns:
        - replace_subs: dict of substitutions
    """
    edges = [(i, j) for i, j in permutations(replace, 2) if i[1].has(j[0])]
    replace_subs = topological_sort([replace, edges], default_sort_key)
    return replace_subs


def cse_subs(cse, state_subs, input_subs, cov_subs):
    replace, expr = cse
    expr = expr[0]

    replace_subs = sort_recursive_subs(replace)

    for (var, sub) in tqdm(replace_subs):
        expr = expr.subs(var, sub)

    print("here)")
    expr = expr.subs(state_subs).subs(input_subs).subs(cov_subs).\
            subs(var_subs).subs(dt, dt_)

    # fin_expr = expr[0].subs(rep_subs)
    # print(fin_expr)
    # quit()
    return expr

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


def quat_mult(p, q):
    r = Matrix([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]])
    return r


def create_cov_matrix(i, j):
    if j >= i:
        # return Symbol("P(" + str(i) + "," + str(j) + ")", real=True)
        # legacy array format
        return Symbol("P[" + str(i) + "][" + str(j) + "]", real=True)
    else:
        return 0


def create_symmetric_cov_matrix(n):
    # define a symbolic covariance matrix
    P = Matrix(n, n, create_cov_matrix)
    for index in range(n):
        for j in range(n):
            if index > j:
                P[index, j] = P[j, index]
    return P


def generate_kalman_gain_equations(P, state, observation, variance, varname: str = "K"):
    H = Matrix([observation]).jacobian(state)
    innov_var = H * P * H.T + Matrix([variance])
    assert (innov_var.shape[0] == 1)
    assert (innov_var.shape[1] == 1)
    K = (P * H.T) / innov_var[0, 0]
    K_simple = cse(K, symbols(f"{varname}0:1000"), optimizations="basic")
    return K_simple


################################################################
################################################################
################################################################


dt = Symbol('dt', real=True)

##################################################
# States
##################################################
# quat = Matrix(MatrixSymbol('quat', 4, 1), real=True)
# pos = Matrix(MatrixSymbol('pos', 3, 1), real=True)
# vel = Matrix(MatrixSymbol('vel', 3, 1), real=True)
# gyro_bias = Matrix(MatrixSymbol('gyro_bias', 3, 1), real=True)
# accel_bias = Matrix(MatrixSymbol('accel_bias', 3, 1), real=True)

# acc_n, acc_e, acc_d = symbols("acc_n, acc_e, acc_d", real=True)
# angvel_n, angvel_e, angvel_d = symbols("angvel_n, angvel_e, angvel_d", real=True)
# acc = Matrix([acc_n, acc_e, acc_d])
# angvel = Matrix([angvel_n, angvel_e, angvel_d])

q0, q1, q2, q3 = symbols("q0, q1, q2, q3", real=True)
pos_n, pos_e, pos_d = symbols("pos_n, pos_e, pos_d", real=True)
vel_n, vel_e, vel_d = symbols("vel_n, vel_e, vel_d", real=True)
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
gyro = Matrix(MatrixSymbol('gyro', 3, 1))
accel = Matrix(MatrixSymbol('accel', 3, 1))
gps_pos = Matrix(MatrixSymbol('gps_pos', 3, 1))
gps_vel = Matrix(MatrixSymbol('gps_vel', 3, 1))

angvel_true = gyro - angvel_bias
acc_true = accel - acc_bias

dcm_to_earth = quat_to_dcm(quat)
dcm_to_body = dcm_to_earth.T
gravity_ef = Matrix([0, 0, -1])  # gravity earth-frame
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
pos_new = pos + vel * dt
vel_new = vel + (dcm_to_earth * (acc_true - gravity_bf)) * dt
gyro_bias_new = angvel_bias
accel_bias_new = acc_bias

##################################################
# Process Noise
##################################################

gyro_x_var, gyro_y_var, gyro_z_var = symbols('gyro_x_var gyro_y_var gyro_z_var')
accel_x_var, accel_y_var, accel_z_var = symbols('accel_x_var accel_y_var accel_z_var')
gps_pos_x_var, gps_pos_y_var, gps_pos_z_var = symbols('acc_pos_x_var acc_pos_y_var acc_pos_z_var')
gps_vel_x_var, gps_vel_y_var, gps_vel_z_var = symbols('acc_vel_x_var acc_vel_y_var acc_vel_z_var')

Q = Matrix.diag([gyro_x_var, gyro_y_var, gyro_z_var,
                 accel_x_var, accel_y_var, accel_z_var])
                 # gps_pos_x_var, gps_pos_y_var, gps_pos_z_var,
                 # gps_vel_x_var, gps_vel_y_var, gps_vel_z_var])

##################################################
# State Prediction
##################################################

state_new = Matrix([quat_new, pos_new, vel_new, accel_bias_new, gyro_bias_new])
input = Matrix([gyro, accel])
A = state_new.jacobian(state)
G = state_new.jacobian(input)

X_new = A * state + G * input

##################################################
# Covariance Prediction
##################################################

P = create_symmetric_cov_matrix(len(state))
P_new = A * P * A.T + G * Q * G.T

for index in range(P.shape[0]):
    for j in range(P.shape[0]):
        if index > j:
            P_new[index,j] = 0

##################################################
# Observation Noise
##################################################

R_accel_x, R_accel_y, R_accel_z = symbols('R_accel_x R_accel_y R_accel_z')
R_mag_world_x, R_mag_world_y, R_mag_world_z = symbols('R_mag_world_x R_mag_world_y R_mag_world_z')
R_gps_pos_x, R_gps_pos_y, R_gps_pos_z = symbols('R_gps_pos_x R_gps_pos_y R_gps_pos_z')
R_gps_vel_x, R_gps_vel_y, R_gps_vel_z = symbols('R_gps_vel_x R_gps_vel_y R_gps_vel_z')

##################################################
# Observations
##################################################

# Body Frame Accelerometer Observation
obs_accel = (dcm_to_earth * gravity_bf) + acc_bias
H_accel = obs_accel.jacobian(state)

# Gps-position Observation (NED frame)
obs_gps_pos = pos
H_gps_pos = obs_gps_pos.jacobian(state)

# Gps-velocity Observation (NED frame)
obs_gps_vel = vel
H_gps_vel = obs_gps_vel.jacobian(state)

##################################################
# Kalman Gains
##################################################

# Accelerometer
innov_var_accel_x = H_accel[0, :] * P * H_accel[0, :].T + Matrix([R_accel_x])
innov_var_accel_y = H_accel[1, :] * P * H_accel[1, :].T + Matrix([R_accel_y])
innov_var_accel_z = H_accel[2, :] * P * H_accel[2, :].T + Matrix([R_accel_z])
K_accel_x = P * H_accel[0, :].T / innov_var_accel_x[0, 0]
K_accel_y = P * H_accel[1, :].T / innov_var_accel_y[0, 0]
K_accel_z = P * H_accel[2, :].T / innov_var_accel_z[0, 0]

# Gps-position
innov_var_gps_pos_x = H_gps_pos[0, :] * P * H_gps_pos[0, :].T + Matrix([R_gps_pos_x])
innov_var_gps_pos_y = H_gps_pos[1, :] * P * H_gps_pos[1, :].T + Matrix([R_gps_pos_y])
innov_var_gps_pos_z = H_gps_pos[2, :] * P * H_gps_pos[2, :].T + Matrix([R_gps_pos_z])
K_gps_pos_x = P * H_gps_pos[0, :].T / innov_var_gps_pos_x[0, 0]
K_gps_pos_y = P * H_gps_pos[1, :].T / innov_var_gps_pos_y[0, 0]
K_gps_pos_z = P * H_gps_pos[2, :].T / innov_var_gps_pos_z[0, 0]

# Gps-velocity
innov_var_gps_vel_x = H_gps_vel[0, :] * P * H_gps_vel[0, :].T + Matrix([R_gps_vel_x])
innov_var_gps_vel_y = H_gps_vel[1, :] * P * H_gps_vel[1, :].T + Matrix([R_gps_vel_y])
innov_var_gps_vel_z = H_gps_vel[2, :] * P * H_gps_vel[2, :].T + Matrix([R_gps_vel_z])
K_gps_vel_x = P * H_gps_vel[0, :].T / innov_var_gps_vel_x[0, 0]
K_gps_vel_y = P * H_gps_vel[1, :].T / innov_var_gps_vel_y[0, 0]
K_gps_vel_z = P * H_gps_vel[2, :].T / innov_var_gps_vel_z[0, 0]

# K = Matrix()
# K = K.col_insert(0 , K_accel_x)
# K = K.col_insert(1 , K_accel_y)
# K = K.col_insert(2 , K_accel_z)
# K = K.col_insert(3 , K_gps_pos_x)
# K = K.col_insert(4 , K_gps_pos_y)
# K = K.col_insert(5 , K_gps_pos_z)
# K = K.col_insert(6 , K_gps_vel_x)
# K = K.col_insert(7 , K_gps_vel_y)
# K = K.col_insert(8 , K_gps_vel_z)

##################################################
# Measurements
##################################################

z_accel_x, z_accel_y, z_accel_z = symbols("z_accel_x, z_accel_y, z_accel_z", real=True)
z_gps_pos_x, z_gps_pos_y, z_gps_pos_z = symbols("z_gps_pos_x, z_gps_pos_y, z_gps_pos_z", real=True)
z_gps_vel_x, z_gps_vel_y, z_gps_vel_z = symbols("z_gps_vel_x, z_gps_vel_y, z_gps_vel_z", real=True)

z_accel = Matrix([z_accel_x, z_accel_y, z_accel_z])
z_gps_pos = Matrix([z_gps_pos_x, z_gps_pos_y, z_gps_pos_z])
z_gps_vel = Matrix([z_gps_vel_x, z_gps_vel_y, z_gps_vel_z])

##################################################
# Updates
##################################################

K_accel = Matrix()
K_accel = K_accel.col_insert(0, K_accel_x)
K_accel = K_accel.col_insert(1, K_accel_y)
K_accel = K_accel.col_insert(2, K_accel_z)

K_gps_pos = Matrix()
K_gps_pos = K_gps_pos.col_insert(0, K_gps_pos_x)
K_gps_pos = K_gps_pos.col_insert(1, K_gps_pos_y)
K_gps_pos = K_gps_pos.col_insert(2, K_gps_pos_z)

K_gps_vel = Matrix()
K_gps_vel = K_gps_vel.col_insert(0, K_gps_vel_x)
K_gps_vel = K_gps_vel.col_insert(1, K_gps_vel_y)
K_gps_vel = K_gps_vel.col_insert(2, K_gps_vel_z)

X_accel_update = K_accel * (z_accel - H_accel * state)
X_gps_pos_update = K_gps_pos * (z_accel - H_gps_pos * state)
X_gps_vel_update = K_gps_vel * (z_accel - H_gps_vel * state)

##################################################
# Write to File
##################################################

# print('Simplifying covariance propagation ...')
# P_new_simple = cse(P_new, symbols("PS0:1000"), optimizations='basic')

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
# return_args = ["nextP"]

# print('Writing covariance propagation to file ...')
# cov_code_generator = PyCodeGenerator("./derivation/nav/generated/cov_predict.py")
# cov_code_generator.print_string("Equations for covariance matrix prediction, without process noise!")
# cov_code_generator.write_function_definition(name="cov_predict",
#                                              args=args)
# cov_code_generator.write_subexpressions(P_new_simple[0])
# cov_code_generator.write_matrix(matrix=Matrix(P_new_simple[1]),
#                                 variable_name="nextP",
#                                 is_symmetric=True)
# cov_code_generator.write_function_returns(returns=return_args)

# cov_code_generator.close()

##################################################
# Sim
##################################################

dt_ = 0.1

state_subs = {
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

input_subs = {
        gyro[0, 0]: 0.0,        # type:ignore
        gyro[1, 0]: 0.0,        # type:ignore
        gyro[2, 0]: 0.0,        # type:ignore
        accel[0, 0]: 0.0,        # type:ignore
        accel[1, 0]: 0.0,        # type:ignore
        accel[2, 0]: 0.0,        # type:ignore
        }

# cov_subs = {var: val for var, val in zip(P, P_init.flatten())}  # type:ignore

var_subs = {
        gyro_x_var: 0,
        gyro_y_var: 0,
        gyro_z_var: 0,
        accel_x_var: 0,
        accel_y_var: 0,
        accel_z_var: 0,
        }

noise_subs = {
        R_accel_x: 0,
        R_accel_y: 0,
        R_accel_z: 0,
        R_gps_pos_x: 0,
        R_gps_pos_y: 0,
        R_gps_pos_z: 0,
        R_gps_vel_x: 0,
        R_gps_vel_y: 0,
        R_gps_vel_z: 0,
        }

meas_subs = {
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

#################################################
# Sympy lambdify funcs
#################################################

vars_X = [
        list(state_subs.keys()),
        list(input_subs.keys()),
        list(get_mat_upper(P)),
        # list(cov_subs.keys()),
        # list(var_subs.keys()),
        dt,
        ]

vars_P = [
        list(state_subs.keys()),
        list(input_subs.keys()),
        list(get_mat_upper(P)),
        # list(cov_subs.keys()),
        list(var_subs.keys()),
        dt,
        ]

vars_all = [
        list(state_subs.keys()),
        list(input_subs.keys()),
        list(get_mat_upper(P)),
        # list(cov_subs.keys()),
        list(var_subs.keys()),
        list(noise_subs.keys()),
        dt,
        ]

X_new_func = sp.lambdify(vars_X, X_new, cse=True)

# covariance
P_new_func = sp.lambdify(vars_P, P_new, cse=True)

# kalman gain
# K_new_func = sp.lambdify(vars_K, K, cse=True)

# update from accel
X_accel_update_func = sp.lambdify(vars_all, X_accel_update, cse=True)

#################################################

# def state_predict(state_subs, input_subs):
    # X = X_new.subs(state_subs).subs(input_subs).subs(dt, dt_)
    # update_subs(state_subs, X)

def state_predict(X, U, P, *args):
    X_new = X_new_func(X, U, P.flatten(), *args)
    return X_new.flatten()

def cov_predict(X, U, P, variance, *args):
    P_new = P_new_func(X, U, P.flatten(), variance, *args)
    return P_new


# def kalman_gain_update(X, U, P, variance, noise):
#     K_new = K_new_func(X, U, P.flatten(), variance, noise, dt_)
#     return K_new


def accel_update(X, U, P, variance, noise, *args):
    X_accel_update_new = X_accel_update_func(X, U, P, variance, noise, *args)
    return X_accel_update_new


def state_update(X, U, P, variance, noise):
    pass


# init
X_init = np.array(list(state_subs.values()))
P_init = np.eye(P.shape[0])
X = X_init
P = P_init

for i in range(100):

    variance = np.array(list(var_subs.values()))
    noise = np.array(list(noise_subs.values()))
    U = np.array(list(input_subs.values()))

    X = state_predict(X, U, get_mat_upper(P), dt_)
    P = cov_predict(X, U, get_mat_upper(P), variance, dt_)

    # K = kalman_gain_update(X, U, get_mat_upper(P), variance, noise)
    X = accel_update(X, U, get_mat_upper(P), variance, noise, dt_)
    print(X)
    quit()

    q = X[:4]
    p = X[4:7]
    v = X[7:10]
    b_gyr = X[10:13]
    b_acc = X[13:16]
    # print(f"q:{q}", f"p:{p}", f"v:{v}", f"b_gyr:{b_gyr}", f"b_acc:{b_acc}")
    # mat_print(P)

