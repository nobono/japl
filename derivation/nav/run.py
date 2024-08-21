import dill as pickle
import numpy as np
from nav_gen import dt, state, input
from nav_gen import dt_
from nav_gen import vars_X, vars_P, vars_all, vars_update
from nav_gen import state_subs, input_subs, var_subs, noise_subs, meas_subs
from nav_gen import get_mat_upper
from nav_gen import array_print
from japl import Sim
from japl import Model
from japl import SimObject
from japl import PyQtGraphPlotter

##################################################
# Setup
##################################################

np.set_printoptions(precision=5, formatter={'float_kind': lambda x: f"{x:5.4f}"})
rand = np.random.uniform
NSTATE = 16

# load
with open("X_new_func.pickle", "rb") as f:
    X_new_func = pickle.load(f)
with open("P_new_func.pickle", "rb") as f:
    P_new_func = pickle.load(f)
with open("X_accel_update_func.pickle", "rb") as f:
    X_accel_update_func = pickle.load(f)
with open("P_accel_update_func.pickle", "rb") as f:
    P_accel_update_func = pickle.load(f)
with open("X_gps_update_func.pickle", "rb") as f:
    X_gps_update_func = pickle.load(f)
with open("P_gps_update_func.pickle", "rb") as f:
    P_gps_update_func = pickle.load(f)

##################################################
# EKF Methods
##################################################


def state_predict(X, U, P, *args):
    X_new = X_new_func(X, U, P.flatten(), *args)
    return X_new.flatten()


def cov_predict(X, U, P, variance, *args):
    P_new = P_new_func(X, U, P.flatten(), variance, *args)
    return P_new


def accel_meas_update(X, U, P, variance, noise, meas, *args):
    X_accel_update_new = X_accel_update_func(X, U, P.flatten(), variance, noise, meas, *args)
    P_accel_update_new = P_accel_update_func(X, U, P.flatten(), variance, noise, meas, *args)
    return X_accel_update_new.flatten(), P_accel_update_new


def gps_meas_update(X, U, P, variance, noise, meas, *args):
    X_gps_update_new = X_gps_update_func(X, U, P.flatten(), variance, noise, meas, *args)
    P_gps_update_new = P_gps_update_func(X, U, P.flatten(), variance, noise, meas, *args)
    return X_gps_update_new.flatten(), P_gps_update_new


##################################################
# Sim
##################################################

# init
X_init = np.array(list(state_subs.values()))
P_init = np.eye(NSTATE)
X = X_init
P = P_init


def ekf_step(t, X, U, dt):
    global P
    variance = np.array(list(var_subs.values()))
    noise = np.array(list(noise_subs.values()))
    meas = np.array(list(meas_subs.values()))
    # meas = np.array([rand(-.02, .02),
    #                  rand(-.02, .02),
    #                  rand(-.02, .02),
    #                  0, 0, 0,
    #                  0, 0, 0])

    U_gyro = np.array([0, 0, 0], dtype=float)
    # U_accel = np.array([0, 0, 0], dtype=float)
    U_accel = rand(-.01, .01, size=3)
    U = np.concatenate([U_gyro, U_accel])

    X = state_predict(X, U, get_mat_upper(P), dt)
    q = X[:4].copy()
    X[:4] = q / np.linalg.norm(q)
    P = cov_predict(X, U, get_mat_upper(P), variance, dt)

    X, P = accel_meas_update(X, U, get_mat_upper(P), variance, noise, meas, dt_)
    # X, P = gps_meas_update(X, U, get_mat_upper(P), variance, noise, meas, dt_)

    q = X[:4]
    p = X[4:7]
    v = X[7:10]
    b_acc = X[10:13]
    b_gyr = X[13:16]
    print(f"q:{q}",
          f"p:{p}",
          f"v:{v}",
          f"b_acc:{b_acc}",
          f"b_gyr:{b_gyr}")
    # print(P)
    # array_print(X)
    return X


plotter = PyQtGraphPlotter(frame_rate=30, figsize=[10, 6], aspect="auto")

print("Building Model...")
model = Model.from_function(dt, state, input, update_func=ekf_step)
simobj = SimObject(model)
simobj.init_state(X)
simobj.plot.set_config({
    "pos-x": {
        "xaxis": 'pos_n',
        "yaxis": 'pos_e'
        },
    # "pos-y": {
    #     "xaxis": 'time',
    #     "yaxis": 'pos_e'
    #     }
    })

print("Starting Sim...")
sim = Sim([0, 10], 0.02, [simobj])
# sim.run()
plotter.animate(sim)
plotter.show()
# quit()
