import os
import dill
import numpy as np
from japl import Sim
from japl import SimObject
from japl import PyQtGraphPlotter
from japl.Library.Earth.Earth import Earth
from japl.Math import Rotation
# from japl.Library.Vehicles.MissileGenericMMD import model

DIR = os.path.dirname(__file__)


with open(f"{DIR}/mmd.pickle", 'rb') as f:
    model = dill.load(f)

plotter = PyQtGraphPlotter(frame_rate=30,
                           figsize=[10, 10],
                           aspect="auto")

ecef0 = [Earth.radius_equatorial, 0, 0]
simobj = SimObject(model)

r0_enu = [0, 0, 0]
v0_enu = [0, 0, 0]
a0_enu = [0, 0, 0]

########################################################
quat0 = [1, 0, 0, 0]
r0_ecef = Rotation.enu_to_ecef_position(r0_enu, ecef0)
v0_ecef = Rotation.enu_to_ecef(v0_enu, ecef0)
r0_eci = Rotation.ecef_to_eci_position(r0_ecef, t=0)
v0_eci = Rotation.ecef_to_eci(v0_ecef, r_ecef=r0_ecef)
alpha_state_0 = [0, 0]
beta_state_0 = [0, 0]
p0 = 0
q0 = 0
r0 = 0
mass0 = 600
mach0 = np.linalg.norm(v0_ecef) / 343.0  # based on ECEF-frame
vel_mag0 = np.linalg.norm(v0_ecef)  # based on ECEF-frame
thrust0 = 0
v0_ecef = v0_ecef

simobj.init_state([quat0, r0_eci, v0_eci,
                   alpha_state_0, beta_state_0,
                   p0,  # q0, r0,
                   mass0,
                   r0_enu, v0_enu, a0_enu,
                   mach0,
                   vel_mag0,
                   v0_ecef,
                   ])
# print(v0_eci)
# print(v0_ecef)
# print(v0_enu)
# print(vel_mag0)
# print('-' * 50)
# quit()
########################################################

parr = ['v_e_x',
        'v_e_y',
        'v_e_z']
parr = ['v_e',
        'v_n',
        'v_u']

simobj.plot.set_config({
    "E": {
        "xaxis": 't',
        "yaxis": parr[0],
        "size": 1,
        },
    "N": {
        "xaxis": 't',
        "yaxis": parr[1],
        "size": 1,
        },
    "U": {
        "xaxis": 't',
        "yaxis": parr[2],
        "size": 1,
        },
    # "N-U": {
    #     "xaxis": 'r_n',
    #     "yaxis": 'r_u',
    #     "size": 2,
    #     },
    "Mach": {
        "xaxis": 't',
        "yaxis": 'mach',
        "size": 1,
        },
    })

sim = Sim(t_span=[0, 10],
          dt=0.01,
          simobjs=[simobj],
          integrate_method="rk4")
# sim.run()
plotter.animate(sim)
plotter.show()
# plotter.plot_obj(simobj)
# plotter.plot(sim.T, simobj.Y[:, 20])
# plotter.plot(simobj.Y[:, 18], simobj.Y[:, 20])
# plotter.plot(simobj.Y[:, 18], simobj.Y[:, 19])
# plotter.plot(simobj.Y[:, 4], simobj.Y[:, 6])

ii = sim.istep
X = simobj.Y[ii]
quat = simobj.get_state_array(X, ["q_0", "q_1", "q_2", "q_3"])
yaw_pitch_roll = np.degrees(Rotation.dcm_to_tait_bryan(Rotation.quat_to_dcm(quat)))

print("quat norm:", np.linalg.norm(quat))
print("quat:", quat)
print("tait_bryan:", yaw_pitch_roll)
print("r_i:", simobj.get_state_array(X, ["r_i_x", "r_i_y", "r_i_z"]))
print("v_i:", simobj.get_state_array(X, ["v_i_x", "v_i_y", "v_i_z"]))
print("alpha:", np.degrees(simobj.get_state_array(X, "alpha")), end=", ")
print("alpha_dot:", np.degrees(simobj.get_state_array(X, "alpha_dot")))
print("beta:", simobj.get_state_array(X, "beta"), end=", ")
print("beta_dot:", simobj.get_state_array(X, "beta_dot"))
print("p:", simobj.get_state_array(X, "p"))
print("mass:", simobj.get_state_array(X, "mass"))
print("r_enu:", simobj.get_state_array(X, ["r_e", "r_n", "r_u"]))
print("v_enu:", simobj.get_state_array(X, ["v_e", "v_n", "v_u"]))
print("a_enu:", simobj.get_state_array(X, ["a_e", "a_n", "a_u"]))
print("mach:", simobj.get_state_array(X, "mach"))
print("vel_mag_ecef:", simobj.get_state_array(X, "vel_mag_ecef"))

# print([np.linalg.norm(i) for i in simobj.Y[:10, -5:-2]])
