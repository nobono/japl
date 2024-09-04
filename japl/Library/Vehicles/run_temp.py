import os
import dill
import numpy as np
from japl import Sim
from japl import SimObject
from japl import PyQtGraphPlotter
from japl.Library.Earth.Earth import Earth
from japl.Math.Rotation import eci_to_enu
from japl.Math.Rotation import eci_to_ecef
from japl.Math.Rotation import enu_to_ecef

DIR = os.path.dirname(__file__)



with open(f"{DIR}/mmd.pickle", 'rb') as f:
    model = dill.load(f)

ecef0 = [Earth.radius_equatorial, 0, 0]
simobj = SimObject(model)


r_enu0 = [0, 0, 1000]
v_enu0 = [0, 400, 200]
a_enu0 = [0, 0, 0]

quat0 = [1, 0, 0, 0]
rm0 = enu_to_ecef(r_enu0, ecef0)
vm0 = enu_to_ecef(v_enu0, ecef0, is_position=False)
alpha_state_0 = [0, 0]
beta_state_0 = [0, 0]
p0 = 0
q0 = 0
r0 = 0
mass0 = 600
mach0 = np.linalg.norm(vm0) / 343.0
vel_mag0 = np.linalg.norm(vm0)

simobj.init_state([quat0, rm0, vm0,
                   alpha_state_0, beta_state_0,
                   p0, q0, r0,
                   mass0,
                   r_enu0, v_enu0, a_enu0,
                   mach0,
                   vel_mag0])

plotter = PyQtGraphPlotter(frame_rate=30,
                           figsize=[10, 6],
                           aspect="auto",
                           # xlim=[0, 20],
                           )

sim = Sim(t_span=[0, 10], dt=0.01, simobjs=[simobj],
          integrate_method="rk4")

# print(simobj.Y[0, 19:22], np.linalg.norm(simobj.Y[0, 19:22]) / 343)

simobj.plot.set_config({
    # "E": {
    #     "xaxis": 'r_e',
    #     "yaxis": 'r_n',
    #     "size": 2,
    #     },
    # "N": {
    #     "xaxis": 't',
    #     "yaxis": 'r_n',
    #     "size": 2,
    #     },
    "U": {
        "xaxis": 'r_n',
        "yaxis": 'r_u',
        "size": 2,
        },
    "M": {
        "xaxis": 't',
        "yaxis": 'vel_mag',
        "size": 2,
        },
    # "U": {
    #     "xaxis": 't',
    #     "yaxis": 'r_u'
    #     },
    # "EN": {
    #     "xaxis": 'r_e',
    #     "yaxis": 'r_n',
    #     },
    # "EU": {
    #     "xaxis": 'r_e',
    #     "yaxis": 'r_u'
    #     },
    # "E": {
    #     "xaxis": 't',
    #     "yaxis": 'r_e'
    #     },
    # "N": {
    #     "xaxis": 't',
    #     "yaxis": 'r_n'
    #     },
    })

sim.run()
# plotter.animate(sim)
# plotter.plot_obj(simobj)
# plotter.plot(sim.T, simobj.Y[:, 20])
# plotter.plot(simobj.Y[:, 18], simobj.Y[:, 20])
# plotter.plot(simobj.Y[:, 18], simobj.Y[:, 19])
# plotter.plot(simobj.Y[:, 4], simobj.Y[:, 6])
# plotter.show()

ii = sim.istep
X = simobj.Y[ii]
quat = simobj.get_state_array(X, ["q_0", "q_1", "q_2", "q_3"])
print("quat norm:", np.linalg.norm(quat))
print("quat:", quat)
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
print("vel_mag:", simobj.get_state_array(X, "vel_mag"))
