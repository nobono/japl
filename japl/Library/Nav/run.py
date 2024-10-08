import dill as pickle
import numpy as np
from japl import Sim
from japl import Model
from japl import SimObject
from japl import PyQtGraphPlotter
from japl.Library.Nav.Nav import dt, state, input
from japl.Library.Nav.Nav import state_info, input_info
from japl.Library.Nav.Nav import variance_info, noise_info
from japl.Library.Nav.Nav import get_mat_upper
from japl.Library.Nav.Nav import accel_var, gyro_var
from japl.Library.Nav.Nav import gps_pos_var, gps_vel_var
from japl.Library.Sensors.OnboardSensors.ImuModel import ImuSensor
from japl.Library.Sensors.OnboardSensors.ImuModel import SensorBase
from japl import JAPL_HOME_DIR

##################################################
# Setup
##################################################

np.set_printoptions(precision=5, formatter={'float_kind': lambda x: f"{x:5.4f}"})
rand = np.random.normal

NSTATE = 16

# load
path = f"{JAPL_HOME_DIR}/data/"
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


def state_predict(X, U, P, variance, noise, *args):
    X_new = X_new_func(X, U, P.flatten(), variance, noise, *args)
    return X_new.flatten()


def cov_predict(X, U, P, variance, noise, *args):
    P_new = P_new_func(X, U, P.flatten(), variance, noise, *args)
    return P_new


def accel_meas_update(X, U, P, variance, noise, *args):
    X_accel_update_new = X_accel_update_func(X, U, P.flatten(), variance, noise, *args)
    P_accel_update_new = P_accel_update_func(X, U, P.flatten(), variance, noise, *args)
    return X_accel_update_new.flatten(), P_accel_update_new


def gps_meas_update(X, U, P, variance, noise, *args):
    X_gps_update_new = X_gps_update_func(X, U, P.flatten(), variance, noise, *args)
    P_gps_update_new = P_gps_update_func(X, U, P.flatten(), variance, noise, *args)
    return X_gps_update_new.flatten(), P_gps_update_new


##################################################
# Sim
##################################################
np.random.seed(1234)

# init
X_init = np.array(list(state_info.values()))
# NOTE: init small as we are certain in inital bias values
P_init = np.eye(NSTATE) * 1e-6

# initialize bias uncertainties high
# pos_x_id = 4
# pos_y_id = 5
# pos_z_id = 6
# vel_x_id = 7
# vel_y_id = 8
# vel_z_id = 9
# accel_bias_x_id = 10
# accel_bias_y_id = 11
# accel_bias_z_id = 12
# gyro_bias_x_id = 13
# gyro_bias_y_id = 14
# gyro_bias_z_id = 15
# P_init[pos_x_id][pos_x_id] = 1e-6
# P_init[pos_y_id][pos_y_id] = 1e-6
# P_init[pos_z_id][pos_z_id] = 1e-6
# P_init[vel_x_id][vel_x_id] = 1e-6
# P_init[vel_y_id][vel_y_id] = 1e-6
# P_init[vel_z_id][vel_z_id] = 1e-6
# P_init[accel_bias_x_id][accel_bias_x_id] = 1e-6
# P_init[accel_bias_y_id][accel_bias_y_id] = 1e-6
# P_init[accel_bias_z_id][accel_bias_z_id] = 1e-1
# P_init[gyro_bias_x_id][gyro_bias_x_id] = 1e-6
# P_init[gyro_bias_y_id][gyro_bias_y_id] = 1e-6
# P_init[gyro_bias_z_id][gyro_bias_z_id] = 1e-6

X = X_init
P = P_init

count = 0

Racc_noise = [accel_var] * 3
Rgyr_noise = [gyro_var] * 3
Rgps_pos_noise = [gps_pos_var] * 3
Rgps_vel_noise = [gps_vel_var] * 3

gyro = SensorBase(noise=gyro_var)
accel = SensorBase(noise=accel_var)
imu = ImuSensor(gyro=gyro, accel=accel)


def ekf_input_update(t, X, U, S, dt):
    global imu
    imu.update(t, [0, 0, 0], [0, 0, 0], [0, 0, 0])
    U_accel = imu.accelerometer.get_latest_measurement()
    U_gyro = imu.gyroscope.get_latest_measurement()
    U_gps_pos = np.array([0, 0, 0])
    U_gps_vel = np.array([0, 0, 0])
    U = np.concatenate([U_gyro.value, U_accel.value, U_gps_pos, U_gps_vel])
    return U


def ekf_step(t, X, U, S, dt):
    global count
    global P
    global imu

    variance = np.array(list(variance_info.values()))
    noise = np.array(list(noise_info.values()))

    X = state_predict(X, U, get_mat_upper(P), variance, noise, dt)
    q = X[:4].copy()
    X[:4] = q / np.linalg.norm(q)

    P = cov_predict(X, U, get_mat_upper(P), variance, noise, dt)

    X, P = accel_meas_update(X, U, get_mat_upper(P), variance, noise, dt)

    sim_hz = (1 / dt)
    gps_hz = 1
    gps_ratio = int(sim_hz / gps_hz)
    if count % gps_ratio == 0:
        X, P = gps_meas_update(X, U, get_mat_upper(P), variance, noise, dt)

    # print(t, np.linalg.norm(P))

    q = X[:4]
    p = X[4:7]
    v = X[7:10]
    b_acc = X[10:13]
    b_gyr = X[13:16]

    gyro = U[:3]
    accel = U[3:6]
    print(
          # f"q:{q}",
          f"p:{p}",
          # f"v:{v}",
          # f"b_acc:{b_acc}",
          # f"b_gyr:{b_gyr}"
          # f"gyro:{gyro}",
          # f"accel:{accel}",
          )
    # print(P)
    # array_print(X)
    count += 1
    return X


# plotter = PyQtGraphPlotter(frame_rate=30, figsize=[10, 8], aspect="auto")

print("Building Model...")
model = Model.from_function(dt, state, input,
                            state_update_func=ekf_step,
                            input_update_func=ekf_input_update)
simobj = SimObject(model)
simobj.init_state(X)
simobj.plot.set_config({

    # "E": {
    #     "xaxis": 'time',
    #     "yaxis": 'pos_e',
    #     "marker": 'o',
    #     },
    # "N": {
    #     "xaxis": 'time',
    #     "yaxis": 'pos_n',
    #     "marker": 'o',
    #     "color": "orange",
    #     },
    # "U": {
    #     "xaxis": 't',
    #     "yaxis": 'pos_d',
    #     "marker": 'o',
    #     "color": "green",
    #     },

    "EU": {
        "xaxis": 'pos_e',
        "yaxis": 'pos_d',
        "marker": 'o',
        # "xlim": [-1, 1],
        # "ylim": [-1, 1]
        },

    # "accel-x": {
    #     "xaxis": 't',
    #     "yaxis": 'z_accel_x',
    #     "marker": 'o',
    #     },
    # "accel-y": {
    #     "xaxis": 't',
    #     "yaxis": 'z_accel_y',
    #     "marker": 'o',
    #     "color": "orange",
    #     },
    # "accel-z": {
    #     "xaxis": 't',
    #     "yaxis": 'z_accel_z',
    #     "marker": 'o',
    #     "color": "green",
    #     },

    })

print("Starting Sim...")
sim = Sim([0, 25], 0.01, [simobj])
sim.run()
sim.profiler.print_info()
# plotter.animate(sim)

iaccel_z = simobj.model.get_input_id("z_accel_z")
ipos_n = simobj.model.get_state_id("pos_n")
ipos_e = simobj.model.get_state_id("pos_e")
ipos_d = simobj.model.get_state_id("pos_d")
T = sim.T
accel_z = simobj.U[:, iaccel_z]
pos_n = simobj.Y[:, ipos_n]
pos_e = simobj.Y[:, ipos_e]
pos_d = simobj.Y[:, ipos_d]

# plotter.plot(T, pos_n)
# plotter.plot(T, pos_e)
# plotter.plot(T, pos_d)
# plotter.show()
