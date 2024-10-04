import dill as pickle
import numpy as np
from nav_gen import dt, state, input
from nav_gen import dt_
from nav_gen import state_subs, input_subs, var_subs, noise_subs, meas_subs
from nav_gen import get_mat_upper
from nav_gen import array_print
from japl import Sim
from japl import Model
from japl import SimObject
from japl import PyQtGraphPlotter
from japl import JAPL_HOME_DIR

##################################################
# Setup
##################################################

np.set_printoptions(precision=5, formatter={'float_kind': lambda x: f"{x:5.4f}"})
# np.random.seed(123)
rand = lambda mag: np.random.uniform(-1, 1) * mag
NSTATE = 16

# load
path = f"{JAPL_HOME_DIR}/derivation/nav/"
with open(path + "X_new_func.pickle", "rb") as f:
    X_new_func = pickle.load(f)
with open(path + "P_new_func.pickle", "rb") as f:
    P_new_func = pickle.load(f)
with open(path + "X_accel_update_func.pickle", "rb") as f:
    X_accel_update_func = pickle.load(f)
with open(path + "P_accel_update_func.pickle", "rb") as f:
    P_accel_update_func = pickle.load(f)
with open(path + "X_gps_update_func.pickle", "rb") as f:
    X_gps_update_func = pickle.load(f)
with open(path + "P_gps_update_func.pickle", "rb") as f:
    P_gps_update_func = pickle.load(f)

##################################################
# EKF Methods
##################################################


def state_predict(X, U, P, variance, noise, meas, *args):
    X_new = X_new_func(X, U, P.flatten(), variance, noise, meas, *args)
    return X_new.flatten()


def cov_predict(X, U, P, variance, noise, meas, *args):
    P_new = P_new_func(X, U, P.flatten(), variance, noise, meas, *args)
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
P_init = np.eye(NSTATE) * 0.3
X = X_init
P = P_init

gps_count = 0


def ekf_step(t, X, U, S, dt):
    global gps_count
    global P
    variance = np.array(list(var_subs.values()))
    noise = np.array(list(noise_subs.values()))

    Racc = 1e-3
    Rgps_pos = .001
    Rgps_vel = .001
    accel_meas = [rand(Racc) for i in range(3)]
    gps_pos_meas = [rand(Rgps_pos) for i in range(3)]
    gps_vel_meas = [rand(Rgps_vel) for i in range(3)]
    # if t > 10 and t < 15:
    #     gps_vel_meas = np.array(gps_vel_meas) + np.array([0, 0, 10])
    meas = np.concatenate([accel_meas, gps_pos_meas, gps_vel_meas])

    # meas = np.array(list(meas_subs.values()))
    # meas = np.array([rand(-.02, .02),
    #                  rand(-.02, .02),
    #                  rand(-.02, .02),
    #                  0, 0, 0,
    #                  0, 0, 0])

    U_gyro = np.array([0, 0, 0], dtype=float)
    # U_accel = np.array([0, 0, 0], dtype=float)
    U_accel = accel_meas
    # U_accel = np.array([rand(-.01, .01),
    #                     rand(-.01, .01),
    #                     rand(-.01, .01) - 1.0,
    #                     ])

    U = np.concatenate([U_gyro, U_accel])

    X = state_predict(X, U, get_mat_upper(P), variance, noise, meas, dt)
    q = X[:4].copy()
    X[:4] = q / np.linalg.norm(q)

    P = cov_predict(X, U, get_mat_upper(P), variance, noise, meas, dt)

    X, P = accel_meas_update(X, U, get_mat_upper(P), variance, noise, meas, dt)

    # for i in range(P.shape[0]):
    #     for j in range(P.shape[0]):
    #         if i > j:
    #             P[i, j] = P[j, i]

    if gps_count % 1 == 0:
        X, P = gps_meas_update(X, U, get_mat_upper(P), variance, noise, meas, dt)

        # for i in range(P.shape[0]):
        #     for j in range(P.shape[0]):
        #         if i > j:
        #             P[i, j] = P[j, i]

    # print(np.linalg.norm(P))

    # q = X[:4]
    # p = X[4:7]
    # v = X[7:10]
    # b_acc = X[10:13]
    # b_gyr = X[13:16]
    # print(f"q:{q}",
    #       f"p:{p}",
    #       f"v:{v}",
    #       f"b_acc:{b_acc}",
    #       f"b_gyr:{b_gyr}")
    # print(P)
    # array_print(X)
    gps_count += 1
    return X


plotter = PyQtGraphPlotter(frame_rate=30, figsize=[10, 6], aspect="auto")

print("Building Model...")
model = Model.from_function(dt, state, input, state_update_func=ekf_step)
simobj = SimObject(model)
simobj.init_state(X)
simobj.plot.set_config({
    "E": {
        "xaxis": 'time',
        "yaxis": 'pos_d',
        "marker": 'o',
        },
    # "E": {
    #     "xaxis": 'pos_e',
    #     "yaxis": 'pos_d',
    #     "marker": 'o',
    #     },
    # "D": {
    #     "xaxis": 'time',
    #     "yaxis": 'pos_d',
    #     "marker": 'o',
        # "xlim": [0, 100],
        # "ylim": [-5, 5]
        # "color": "red",
        # }
    })

print("Starting Sim...")
sim = Sim([0, 150], 0.1, [simobj])
sim.run()
# plotter.animate(sim)
ipos_e = simobj.model.get_state_id("pos_e")
ipos_d = simobj.model.get_state_id("pos_d")
T = sim.T
pos_e = simobj.Y[:, ipos_e]
pos_d = simobj.Y[:, ipos_d]
plotter.plot(T, pos_d)
plotter.show()
# quit()
