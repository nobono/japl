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

q0 = [1, 0, 0, 0]
r0 = enu_to_ecef(r_enu0, ecef0)
v0 = enu_to_ecef(v_enu0, ecef0, is_position=False)
alpha_state_0 = [0, 0]
beta_state_0 = [0, 0]
p0 = 0
mass0 = 600
mach0 = np.linalg.norm(v0) / 343.0

# print(v0)
# print(mach0)
# print(enu_to_ecef([0, 0, 0], ecef0, is_position=False))
# quit()

# Earth-relative velocity vector
# omega_e = Earth.omega
# C_eci_to_ecef = np.array([
#     [np.cos(0), np.sin(0), 0],   # type:ignore
#     [-np.sin(0), np.cos(0), 0],  # type:ignore
#     [0, 0, 1]])
# omega_skew_ie = np.array([
#     [0, -omega_e, 0],
#     [omega_e, 0, 0],
#     [0, 0, 0],
#     ])
# v0 = C_eci_to_ecef @ v0 - omega_skew_ie @ (C_eci_to_ecef @ r0)

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

sim = Sim(t_span=[0, 50], dt=0.01, simobjs=[simobj],
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
print("mach:", simobj.Y[-1, 25])

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
        "yaxis": 'mach',
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
plotter.animate(sim)
# plotter.plot_obj(simobj)
# plotter.plot(sim.T, simobj.Y[:, 20])
# plotter.plot(simobj.Y[:, 18], simobj.Y[:, 20])
# plotter.plot(simobj.Y[:, 18], simobj.Y[:, 19])
# plotter.plot(simobj.Y[:, 4], simobj.Y[:, 6])
plotter.show()