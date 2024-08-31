import os
import dill
import numpy as np
from japl import Sim
from japl import SimObject
from japl import PyQtGraphPlotter
from japl.Library.Earth.Earth import Earth
from japl.Math.Rotation import eci_to_enu

DIR = os.path.dirname(__file__)



dcm_ecef_to_enu = np.array([[-np.sin(0), np.cos(0), 0],
                            [-np.sin(0) * np.cos(0), -np.sin(0) * np.sin(0), np.cos(0)],
                            [np.cos(0) * np.cos(0), np.cos(0) * np.sin(0), np.sin(0)]])


with open(f"{DIR}/mmd.pickle", 'rb') as f:
    model = dill.load(f)

simobj = SimObject(model)
q0 = [1, 0, 0, 0]
r0 = [Earth.radius_equatorial, 0, 0]
v0 = dcm_ecef_to_enu.T @ np.array([0, 300, 600])
alpha_state_0 = [0, 0]
beta_state_0 = [0, 0]
p0 = 0
mass0 = 600
r_enu0 = [0, 0, 0]
v_enu0 = [0, 0, 0]
a_enu0 = [0, 0, 0]
mach0 = np.linalg.norm(v0) / 343.0

# simobj.init_state([q0, r0, v0, alpha0, beta0, p0])
simobj.init_state([q0, r0, v0,
                   alpha_state_0, beta_state_0,
                   p0,
                   mass0,
                   r_enu0, v_enu0, a_enu0,
                   mach0])

plotter = PyQtGraphPlotter(frame_rate=30,
                           figsize=[10, 6],
                           aspect="auto",
                           # xlim=[0, 20],
                           )

sim = Sim(t_span=[0, 100], dt=0.01, simobjs=[simobj],
          integrate_method="rk4")
sim.run()

print("quat:", simobj.Y[-1, :4])
print("quat norm:", np.linalg.norm(simobj.Y[-1, :4]))
print("r_i:", simobj.Y[-1, 4:7])
print("v_i:", simobj.Y[-1, 7:10])
print("alpha_state:", np.degrees(simobj.Y[-1, 10:12]))
print("beta_state:", simobj.Y[-1, 12:14])
print("p:", simobj.Y[-1, 14])
print("mass:", simobj.Y[-1, 15])
print("r_enu:", simobj.Y[-1, 16:19])
print("v_enu:", simobj.Y[-1, 19:22])
print("a_enu:", simobj.Y[-1, 22:25])

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
        "yaxis": 'mach',
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
plotter.animate(sim)
# plotter.plot_obj(simobj)
# plotter.plot(sim.T, simobj.Y[:, 20])
# plotter.plot(simobj.Y[:, 18], simobj.Y[:, 20])
# plotter.plot(simobj.Y[:, 18], simobj.Y[:, 19])
# plotter.plot(simobj.Y[:, 4], simobj.Y[:, 6])
plotter.show()
