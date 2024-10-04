import dill as pickle
import timeit
import numpy as np
import sympy as sp
from tqdm import tqdm
from itertools import permutations
from sympy import Matrix, Symbol, symbols, cse
from sympy import MatrixSymbol
from sympy import default_sort_key, topological_sort
# from code_gen import PyCodeGenerator
from japl.BuildTools.CodeGeneration import OctaveCodeGenerator
# from code_gen import CCodeGenerator
from japl import Model
from japl.BuildTools.DirectUpdate import DirectUpdate
from japl import JAPL_HOME_DIR
from japl.Sim.Sim import Sim
from japl.SimObject.SimObject import SimObject
from japl.Util.Util import flatten_list


################################################################
# Helper Methods
################################################################


def profile(func):
    def wrapped(*args, **kwargs):
        start_time = timeit.default_timer()
        res = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print("-" * 50)
        print(f"{func.__name__} executed in {end_time - start_time} seconds")
        print("-" * 50)
        return res
    return wrapped


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


def zero_mat_lower(mat):
    ret = mat.copy()
    for index in range(mat.shape[0]):
        for j in range(mat.shape[0]):
            if index > j:
                ret[index, j] = 0
    return ret


def mat_print(mat):
    if isinstance(mat, Matrix):
        mat = np.array(mat)
    for row in mat:
        print('[', end="")
        for item in row:
            if item == 0:
                print("%s, " % (' ' * 8), end="")
            else:
                print("%.6f, " % item, end="")
        print(']')
    print()


def mat_print_sparse(mat):
    for i in range(mat.shape[0]):
        print('[', end="")
        for j in range(mat.shape[1]):
            item = mat[i, j]
            try:
                item = float(item)
                if item == 0:
                    print("%s, " % " ", end="")
                else:
                    print("%d, " % 1, end="")
            except:  # noqa
                if isinstance(item, sp.Expr):
                    print("%d, " % 1, end="")
        print(']')
    print()


def array_print(mat):
    print('[', end="")
    for item in mat:
        if item == 0:
            print("%s, " % (' ' * 8), end="")
        else:
            print("%.6f, " % item, end="")
    print(']')


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


def cse_subs(cse, state_subs, input_subs, cov_subs, var_subs):
    replace, expr = cse
    expr = expr[0]
    replace_subs = sort_recursive_subs(replace)
    for (var, sub) in tqdm(replace_subs):
        expr = expr.subs(var, sub)
    expr = expr.subs(state_subs).subs(input_subs).subs(cov_subs).subs(var_subs).subs(dt, dt_)
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


t = Symbol('t', real=True)
dt = Symbol('dt', real=True)

##################################################
# States
##################################################

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
gyro_x, gyro_y, gyro_z = symbols("gyro_x, gyro_y, gyro_z", real=True)
accel_x, accel_y, accel_z = symbols("accel_x, accel_y, accel_z", real=True)
gps_pos_x, gps_pos_y, gps_pos_z = symbols("gps_pos_x, gps_pos_y, gps_pos_z", real=True)
gps_vel_x, gps_vel_y, gps_vel_z = symbols("gps_vel_x, gps_vel_y, gps_vel_z", real=True)

gyro = Matrix([gyro_x, gyro_y, gyro_z])
accel = Matrix([accel_x, accel_y, accel_z])
gps_pos = Matrix([gps_pos_x, gps_pos_y, gps_pos_z])
gps_vel = Matrix([gps_vel_x, gps_vel_y, gps_vel_z])

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

# NOTE: since no magnetometer yet, zero-out gyro_z bias
gyro_bias_new[2] = sp.Float(0)

##################################################
# Process Noise
##################################################

gyro_x_var, gyro_y_var, gyro_z_var = symbols('gyro_x_var gyro_y_var gyro_z_var')
accel_x_var, accel_y_var, accel_z_var = symbols('accel_x_var accel_y_var accel_z_var')
# gps_pos_x_var, gps_pos_y_var, gps_pos_z_var = symbols('gps_pos_x_var, gps_pos_y_var, gps_pos_z_var')

input_var = Matrix.diag([gyro_x_var, gyro_y_var, gyro_z_var,
                         accel_x_var, accel_y_var, accel_z_var])

##################################################
# State Prediction
##################################################

state_new = Matrix([quat_new, pos_new, vel_new, accel_bias_new, gyro_bias_new])
input = Matrix([gyro, accel])
F = state_new.jacobian(state)
G = state_new.jacobian(input)

X_new = F * state + G * input

Q = G * input_var * G.T

##################################################
# Covariance Prediction
##################################################

# P = create_symmetric_cov_matrix(len(state))
P = MatrixSymbol("P", len(state), len(state)).as_mutable()
P_new = F * P * F.T + Q

for index in range(P.shape[0]):
    for j in range(P.shape[0]):
        if index > j:
            P_new[index, j] = 0

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
print("Building Observations...")

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
# Measurements
##################################################

z_accel_x, z_accel_y, z_accel_z = symbols("z_accel_x, z_accel_y, z_accel_z", real=True)
z_gps_pos_x, z_gps_pos_y, z_gps_pos_z = symbols("z_gps_pos_x, z_gps_pos_y, z_gps_pos_z", real=True)
z_gps_vel_x, z_gps_vel_y, z_gps_vel_z = symbols("z_gps_vel_x, z_gps_vel_y, z_gps_vel_z", real=True)

z_accel = Matrix([z_accel_x, z_accel_y, z_accel_z])
# z_gps_pos = Matrix([z_gps_pos_x, z_gps_pos_y, z_gps_pos_z])
# z_gps_vel = Matrix([z_gps_vel_x, z_gps_vel_y, z_gps_vel_z])
z_gps = Matrix([z_gps_pos_x, z_gps_pos_y, z_gps_pos_z,
                z_gps_vel_x, z_gps_vel_y, z_gps_vel_z])
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
K_gps = K_gps.col_insert(0, K_gps_vel_x)
K_gps = K_gps.col_insert(1, K_gps_vel_y)
K_gps = K_gps.col_insert(2, K_gps_vel_z)

# accelerometer
X_accel_update = state + K_accel * (z_accel - H_accel * state)
P_accel_update = P - (K_accel * H_accel * P)

# gps
X_gps_update = state + K_gps * (z_gps - H_gps * state)
P_gps_update = P - (K_gps * H_gps * P)

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

# process noise
var_subs = {
        gyro_x_var: .001,
        gyro_y_var: .001,
        gyro_z_var: .001,
        accel_x_var: .001,
        accel_y_var: .001,
        accel_z_var: .001,
        }

# meas noise
noise_subs = {
        R_accel_x: 0.1,
        R_accel_y: 0.1,
        R_accel_z: 0.1,
        R_gps_pos_x: 0.01,
        R_gps_pos_y: 0.01,
        R_gps_pos_z: 0.01,
        R_gps_vel_x: 0.01,
        R_gps_vel_y: 0.01,
        R_gps_vel_z: 0.01,
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
# TODO:
# TODO:
# TODO:
# TODO: input_subs accel and meas_subs z_accel both
#       existing is a problem because they are the same
#       thing. need to fix this.

#################################################
# Sympy lambdify funcs
#################################################

vars_X = [
        list(state_subs.keys()),
        list(input_subs.keys()),
        list(get_mat_upper(P)),
        dt,
        ]

vars_P = [
        list(state_subs.keys()),
        list(input_subs.keys()),
        list(get_mat_upper(P)),
        list(var_subs.keys()),
        dt,
        ]

vars_all = [
        list(state_subs.keys()),
        list(input_subs.keys()),
        list(get_mat_upper(P)),
        list(var_subs.keys()),
        list(noise_subs.keys()),
        dt,
        ]

vars_update = [
        list(state_subs.keys()),
        list(input_subs.keys()),
        list(get_mat_upper(P)),
        list(var_subs.keys()),
        list(noise_subs.keys()),
        list(meas_subs.keys()),
        dt,
        ]

######################################################
# NOTE: trying to create model from_expression
######################################################
# state_update = Matrix([
#     DirectUpdate(state, X_new),
#     DirectUpdate(P, P_new),
#     ])

# static = Matrix([
#     gyro_x_var,
#     gyro_y_var,
#     gyro_z_var,
#     accel_x_var,
#     accel_y_var,
#     accel_z_var,
#     ])

# # X_new
# # P_new
# # X_accel_update
# # P_accel_update
# # X_gps_update
# # P_gps_update


# # model = Model.from_expression(dt_var=dt,
# #                               state_vars=state_update,
# #                               input_vars=input,
# #                               static_vars=static,
# #                               dynamics_expr=Matrix([np.nan] * len(state)),
# #                               use_multiprocess_build=True)
# # model.save("./data", "ekf")

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

    print("Lambdify-ing Symbolic Expressions...")

    # state predict
    X_new_func = profile(sp.lambdify)(vars_X, X_new, cse=True)

    # covariance predict
    P_new_func = profile(sp.lambdify)(vars_P, P_new, cse=True)

    # update from accel
    X_accel_update_func = profile(sp.lambdify)(vars_update, X_accel_update, cse=True)
    P_accel_update_func = profile(sp.lambdify)(vars_update, P_accel_update, cse=True)

    # update from gps-position
    X_gps_update_func = profile(sp.lambdify)(vars_update, X_gps_update, cse=True)
    P_gps_update_func = profile(sp.lambdify)(vars_update, P_gps_update, cse=True)

    out = [("X_new_func", X_new_func),
           ("P_new_func", P_new_func),
           ("X_accel_update_func", X_accel_update_func),
           ("P_accel_update_func", P_accel_update_func),
           ("X_gps_update_func", X_gps_update_func),
           ("P_gps_update_func", P_gps_update_func)]

    for (name, func) in out:
        print(f"saving {name}...")
        with open(f"{JAPL_HOME_DIR}/derivation/nav/{name}.pickle", "wb") as f:
            pickle.dump(func, f)
