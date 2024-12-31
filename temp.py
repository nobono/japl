import numpy as np
import quaternion
from japl.Library.Earth.Earth import Earth
from japl import Rotation
from japl.Util import parse_yaml
from japl import PyQtGraphPlotter
from japl import SimObject
from japl import Model
from japl import Sim
from japl import AeroTable
from japl import Atmosphere
from japl.global_opts import get_root_dir
from pathlib import Path
from astropy import units as u
from aerotable import AeroTable as CppAeroTable
from datatable import DataTable as CppDataTable
import mmd

####
# aero_file_path = Path(get_root_dir(), "aerodata/cms_sr_stage1aero.mat")
# pyaero = AeroTable(aero_file_path)

# model = mmd.Model()
# model.set_aerotable(pyaero)
# print(model.aerotable.get_Sref())
# print(model.cpp.aerotable.get_Sref())

# simobj = mmd.SimObject()
# simobj.model.set_aerotable(pyaero)
# print(simobj.model.aerotable.get_Sref())
# print(simobj.model.cpp.aerotable.get_Sref())
# quit()
####



simobj = mmd.SimObject()

# aero_file_path = Path(get_root_dir(), "aerodata/cms_sr_stage1aero.mat")
# pyaero = AeroTable(aero_file_path)
# simobj.model.set_aerotable(pyaero)

# print(id(simobj.model.cpp.input_updates))
# simobj.model.set_aerotable(pyaero)
# print(id(simobj.model.input_updates))

# -------------------------------------------------------------------------
# from japl.Library.Vehicles.MissileGenericMMD import (dt,
#                                                      state,
#                                                      input,
#                                                      dynamics,
#                                                      static,
#                                                      modules,
#                                                      defs)
# model = Model.from_expression(dt,
#                               state,
#                               input,
#                               dynamics,
#                               static_vars=static,
#                               modules=modules,
#                               definitions=defs,
#                               use_multiprocess_build=True)
# model.cache_build()

stage1 = AeroTable("./aerodata/stage_1_aero.mat",
                   angle_units=u.deg,
                   length_units=u.imperial.foot)
stage2 = AeroTable("./aerodata/stage_2_aero.mat")

data = np.array(stage1.get_stage().CNB)
axes = stage1.get_stage().CNB.axes
aero = AeroTable()
aero.add_stage(stage1)
aero.add_stage(stage2)
simobj.model.set_aerotable(aero)

# simobj = SimObject(model)

inits = dict(
        q_0=1,
        q_1=0,
        q_2=0,
        q_3=0,
        r_i_x=6_378_137.0,
        r_i_y=0,
        r_i_z=0,
        v_i_x=50,
        v_i_y=50,
        v_i_z=0,
        alpha=0,
        alpha_dot=0,
        beta=0,
        beta_dot=0,
        p=0,
        wet_mass=100,
        dry_mass=50,

        omega_n=50,
        zeta=0.7,
        K_phi=1,
        omega_p=20,
        phi_c=0,
        T_r=0.5,
        is_boosting=0,
        stage=0,
        is_launched=1)
# -------------------------------------------------------------------------


def input_func(*args):
    U = np.array([0., 0, 0, 0, 50, 0, 9.81], dtype=float)
    return U


VLEG = 50
ecef0 = np.array([Earth.radius_equatorial, 0, 0])
r0_enu = np.array([0, 0, 30], dtype=float)
v0_enu = np.array([0, 3, 3], dtype=float)
a0_enu = np.array([0, 0, 0], dtype=float)
wet_mass0 = 108 / 2.2
dry_mass0 = 11.
omega_n = 50  # natural frequency
zeta = 0.7    # damping ratio

K_phi = 1     # roll gain
omega_p = 20  # natural frequency (roll)
phi_c = 0     # roll angle command
T_r = 0.5     # roll autopilot time constant

vleg_ang = np.radians(VLEG - 90.0)
quat0 = [np.cos(vleg_ang / 2), 0, np.sin(vleg_ang / 2), 0]

q_0, q_1, q_2, q_3 = quat0
C_body_to_eci = np.array([
    [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],   # type:ignore # noqa
    [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],   # type:ignore # noqa
    [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa

C_lma = Rotation.euler_to_dcm(*[0., 0., 0.])
quat_lma = quaternion.from_rotation_matrix(C_lma)
quat_launcher = (quat_lma * quaternion.from_float_array(quat0)).components

r0_ecef = Rotation.enu_to_ecef_position(r0_enu, ecef0)
v0_ecef = Rotation.enu_to_ecef(v0_enu, ecef0)
a0_ecef = Rotation.enu_to_ecef(a0_enu, ecef0)
r0_eci = Rotation.ecef_to_eci(r0_ecef, t=0)
v0_eci = Rotation.ecef_to_eci_velocity(v0_ecef, r_ecef=r0_ecef)
a0_eci = Rotation.ecef_to_eci(a0_ecef, t=0)
alpha0 = 0
alpha_dot0 = 0
beta0 = 0
beta_dot0 = 0
phi_hat0 = 0
phi_hat_dot0 = 0
p0 = 0
q0 = 0
r0 = 0
mach0 = np.linalg.norm(v0_ecef) / 343.0  # based on ECEF-frame
vel_mag0 = np.linalg.norm(v0_ecef)  # based on ECEF-frame
vel_mag_dot0 = 0
v0_body = C_body_to_eci.T @ v0_eci
v0_body_hat = v0_body / np.linalg.norm(v0_body)
g0_body = C_body_to_eci.T @ np.array([-9.81, 0, 0])
a0_body = C_body_to_eci.T @ a0_eci
# wet_mass0 = wet_mass0  # mass_props.wet_mass  # + (24.1224 / 2.2)
# dry_mass0 = dry_mass0
lift0 = 0
slip0 = 0
drag0 = 0
CA0 = 0
CNB0 = 0
q_bar0 = 0
rho0 = 0
accel0 = a0_body  # specific force (acceleromter measures this)

# inits = parse_yaml("./mmd/config_state.yaml")
simobj.init_state([quat_launcher,
                   r0_eci, v0_eci, a0_eci,
                   alpha0, alpha_dot0,
                   beta0, beta_dot0,
                   phi_hat0, phi_hat_dot0,
                   p0, q0, r0,
                   r0_enu, v0_enu, a0_enu,
                   r0_ecef, v0_ecef, a0_ecef,
                   vel_mag0,
                   vel_mag_dot0,
                   mach0,
                   v0_body,
                   v0_body_hat,
                   g0_body,
                   a0_body,
                   wet_mass0,
                   dry_mass0,
                   CA0,
                   CNB0,
                   q_bar0,
                   lift0,
                   slip0,
                   drag0,
                   accel0,
                   rho0,
                   ])
omega_n = omega_n
zeta = zeta
K_phi = K_phi
omega_p = omega_p
phi_c = phi_c
T_r = T_r
is_boosting = 0
stage = 1
is_launched = 0

simobj.init_static([
    omega_n,
    zeta,
    K_phi,
    omega_p,
    phi_c,
    T_r,
    is_boosting,
    stage,
    is_launched,
    ])

simobj.set_input_function(input_func)

t = 0.
X = simobj.X0
U = np.array([0., 0, 0, 1, 0, 0, 9.81])
S = simobj.S0
dt = 0.1

# ret = simobj.model.state_updates(t, X, U, S, dt)
# ret = simobj.model.dynamics(t, X, U, S, dt)
# ret = simobj.model.input_updates(t, X, U, S, dt)

simobj.plot.set_config({
    "NU": {"xaxis": simobj.r_n,
           "yaxis": simobj.r_u},
    "velU": {"xaxis": 't',
             "yaxis": simobj.v_u},
    # "EAST": {"xaxis": "time",
    #          "yaxis": simobj.r_e},
    # "NORTH": {"xaxis": "time",
    #           "yaxis": simobj.r_n},
    # "UP": {"xaxis": "time",
    #        "yaxis": simobj.r_u},
    })

sim = Sim(t_span=[0, 100], dt=0.1, simobjs=[simobj])

plotter = PyQtGraphPlotter(figsize=[10, 10],
                           frame_rate=30,
                           aspect="auto",
                           axis_color="grey",
                           background_color="black",
                           antialias=False,
                           # quiet=True,
                           )

plotter.animate(sim).show()
# print("done")
# sim.run()
# sim.profiler.print_info()
# print(simobj.Y)
