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
# np.random.seed(123)

# init
X_init = np.array(list(state_info.values()))
P_init = np.eye(NSTATE) * 0.3
X = X_init
P = P_init

count = 0

accel_Hz = 100
accel_noise_density = 10e-6 * 9.81  # (ug / sqrt(Hz))
Racc = accel_noise_density**2 * accel_Hz
Racc_std = np.sqrt(Racc)

gyro_Hz = 100
gyro_noise_density = 3.5e-6  # ((udeg / s) / sqrt(Hz))
Rgyr = gyro_noise_density**2 * gyro_Hz
Rgyr_std = np.sqrt(Rgyr)

Rgps_pos = 0.1**2
Rgps_pos_std = np.sqrt(Rgps_pos)

Rgps_vel = 0.2**2
Rgps_vel_std = np.sqrt(Rgps_vel)

gyro = SensorBase(noise=[Rgyr_std] * 3)
accel = SensorBase(noise=[Racc_std] * 3)
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
    global Racc, Rgyr
    global imu

    variance = np.array(list(variance_info.values()))
    noise = np.array(list(noise_info.values()))

    # gyro_meas = simobj.get_input_array(U, ["z_gyro_x", "z_gyro_y", "z_gyro_y"])
    # accel_meas = simobj.get_input_array(U, ["z_accel_x", "z_accel_y", "z_accel_y"])
    # gps_pos_meas = simobj.get_input_array(U, ["z_gps_pos_x", "z_gps_pos_y", "z_gps_pos_z"])
    # gps_vel_meas = simobj.get_input_array(U, ["z_gps_vel_x", "z_gps_vel_y", "z_gps_vel_z"])

    # meas = np.concatenate([gyro_meas, accel_meas, gps_pos_meas, gps_vel_meas])

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
    count += 1
    return X


plotter = PyQtGraphPlotter(frame_rate=30, figsize=[10, 8], aspect="auto")

print("Building Model...")
model = Model.from_function(dt, state, input,
                            state_update_func=ekf_step,
                            input_update_func=ekf_input_update)
simobj = SimObject(model)
simobj.init_state(X)
simobj.plot.set_config({
    # "E": {
    #     "xaxis": 'time',
    #     "yaxis": 'pos_d',
    #     "marker": 'o',
    #     },
    "U": {
        "xaxis": 't',
        "yaxis": 'pos_d',
        "marker": 'o',
        },
    # "accel-z": {
    #     "xaxis": 't',
    #     "yaxis": 'z_accel_z',
    #     "marker": 'o',
    #     },
    # "D": {
    #     "xaxis": 'time',
    #     "yaxis": 'pos_d',
    #     "marker": 'o',
    #     "color": "red",
    #     }
    })

print("Starting Sim...")
sim = Sim([0, 200], 0.01, [simobj])
sim.run()
sim.profiler.print_info()
# plotter.animate(sim)

iaccel_z = simobj.model.get_input_id("z_accel_z")
ipos_e = simobj.model.get_state_id("pos_e")
ipos_d = simobj.model.get_state_id("pos_d")
T = sim.T
accel_z = simobj.U[:, iaccel_z]
pos_e = simobj.Y[:, ipos_e]
pos_d = simobj.Y[:, ipos_d]

plotter.plot(T, pos_d)
plotter.show()
