import os
import dill
import numpy as np
import quaternion
from japl import Sim
from japl import SimObject
from japl import PyQtGraphPlotter
from japl.Aero.AeroTable import AeroTable
from japl.Library.Earth.Earth import Earth
from japl.Math import Rotation
from japl import Model
from japl.MassProp.MassPropTable import MassPropTable
from pprint import pprint
from japl.Library.Vehicles.MissileGenericMMD_function import pog

DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=3)



########################################################
# Load Tables
########################################################

mass_props = MassPropTable(DIR + "/../../../aeromodel/stage_1_mass.mat",
                           from_template="CMS")
# aero = AeroTable(DIR + "/../../../aeromodel/stage_1_aero.mat",
#                        from_template="CMS")

########################################################
# Custom Input Function
########################################################


def user_input_func(t, X, U, dt, *args):
    global mass_props
    global complete

    alpha = X[10]
    r_enu_m = X[18:21]
    alt = r_enu_m[2]
    v_enu_m = X[21:24]
    p = X[14]
    q = X[15]
    r = X[16]

    complete, a_c = pog(t,
                        desired_vleg=45,
                        desired_bearing_angle=0,
                        alphaTotal=alpha,
                        altm=alt,
                        lead_angle=0,
                        vm=v_enu_m,
                        body_rates=np.array([p, q, r])
                        )
    if complete:
        pass
    a_c_y = a_c[1]
    a_c_z = a_c[2]

    # omega_e = Earth.omega
    # q_0, q_1, q_2, q_3 = X[:4]
    # r_e_m = X[27:30]
    # C_eci_to_ecef = np.array([
    #     [np.cos(omega_e * t), np.sin(omega_e * t), 0],
    #     [-np.sin(omega_e * t), np.cos(omega_e * t), 0],
    #     [0, 0, 1]])
    # C_body_to_eci = np.array([
    #     [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],  # type:ignore # noqa
    #     [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],  # type:ignore # noqa
    #     [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa
    # C_body_to_ecef = C_eci_to_ecef @ C_body_to_eci
    # lla0 = Rotation.ecef_to_lla(r_e_m)
    # lat0, lon0, _ = lla0
    # C_ecef_to_enu = np.array([
    #     [-np.sin(lon0), np.cos(lon0), 0],
    #     [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
    #     [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
    #     ])
    # acc_cmd_ecef = C_ecef_to_enu.T @ np.array([0, 1, 1])
    # acc_cmd_body = C_body_to_ecef.T @ acc_cmd_ecef
    # acc_cmd_body = 60 * (acc_cmd_body / np.linalg.norm(acc_cmd_body))
    # a_c_y = acc_cmd_body[1]
    # a_c_z = acc_cmd_body[2]
    # if mass_props.get_mass_dot(t) == 0:
    #     pass

    U[2] = a_c_y                       # acc cmd
    U[3] = a_c_z                       # acc cmd
    U[4] = mass_props.get_thrust(t)    # thrust
    U[5] = mass_props.get_mass_dot(t)  # mass_dot


########################################################
# Load model
########################################################

# with open(f"{DIR}/mmd.pickle", 'rb') as f:
#     model: Model = dill.load(f)
model = Model.from_file(DIR + "/mmd.japl")
model.set_input_function(user_input_func)

########################################################
# Setup plotter
########################################################

plotter = PyQtGraphPlotter(frame_rate=30,
                           figsize=[10, 10],
                           aspect="auto",
                           # xlim=[-500, 500],
                           # ylim=[0, 600],
                           # ff=5.0,
                           )

########################################################
# Initialize Model
########################################################

t_span = [0, 3]
ecef0 = [Earth.radius_equatorial, 0, 0]
simobj = SimObject(model)

r0_enu = [0, 0, 0]
v0_enu = [0, 0, 1]
a0_enu = [0, 0, -9.81]
dcm = Rotation.tait_bryan_to_dcm(np.radians([0, 0, 0])).T  # yaw-pitch-roll
quat0 = quaternion.from_rotation_matrix(dcm).components
# quat0 = [1, 0, 0, 0]

q_0, q_1, q_2, q_3 = quat0
C_body_to_eci = np.array([
    [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],   # type:ignore # noqa
    [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],   # type:ignore # noqa
    [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa

########################################################

r0_ecef = Rotation.enu_to_ecef_position(r0_enu, ecef0)
v0_ecef = Rotation.enu_to_ecef(v0_enu, ecef0)
a0_ecef = Rotation.enu_to_ecef(a0_enu, ecef0)
r0_eci = Rotation.ecef_to_eci_position(r0_ecef, t=0)
v0_eci = Rotation.ecef_to_eci(v0_ecef, r_ecef=r0_ecef)
alpha_state_0 = [0, 0]
beta_state_0 = [0, 0]
p0 = 0
q0 = 0
r0 = 0
mach0 = np.linalg.norm(v0_ecef) / 343.0  # based on ECEF-frame
vel_mag0 = np.linalg.norm(v0_ecef)  # based on ECEF-frame
vel_mag0_dot = 0
thrust0 = 0
v0_ecef = v0_ecef
v0_body = C_body_to_eci.T @ v0_eci
v0_body_hat = v0_body / np.linalg.norm(v0_body)
g0_body = [0, 0, -9.81]
a0_body = [0, 0, -9.81]
wet_mass0 = mass_props.wet_mass
dry_mass0 = mass_props.dry_mass
CA0 = 0
CNB0 = 0

simobj.init_state([quat0,
                   r0_eci, v0_eci,
                   alpha_state_0, beta_state_0,
                   p0, q0, r0,
                   r0_enu, v0_enu, a0_enu,
                   r0_ecef, v0_ecef, a0_ecef,
                   vel_mag0,
                   vel_mag0_dot,
                   mach0,
                   v0_body,
                   v0_body_hat,
                   g0_body,
                   a0_body,
                   wet_mass0,
                   dry_mass0,
                   CA0,
                   CNB0,
                   ])

########################################################

# parr = ['v_b_e_x',
#         'v_b_e_y',
#         'v_b_e_z']
parr = ['r_e_x',
        'r_e_y',
        'r_e_z']

# TODO make set_config() a method
# which appends accaptable arguments / dict
simobj.plot.set_config({

    # "E": {"xaxis": 't', "yaxis": parr[0]},
    # "N": {"xaxis": 't', "yaxis": parr[1]},
    # "U": {"xaxis": 't', "yaxis": parr[2]},

    "N-U": {"xaxis": 'r_n', "yaxis": 'r_u'},
    # "N-E": {"xaxis": 'r_n', "yaxis": 'r_e'},

    # "Mach": {"xaxis": 't', "yaxis": 'mach'},
    "Thrust": {"xaxis": 't', "yaxis": 'thrust'},
    # "V": {"xaxis": 't', "yaxis": 'vel_mag_e'},

    # "alpha": {"xaxis": 't', "yaxis": 'alpha'},
    # "beta": {"xaxis": 't', "yaxis": 'beta'},

    # "wet_mass": {"xaxis": 't', "yaxis": 'wet_mass'},
    # "mass_dot": {"xaxis": 't', "yaxis": 'mass_dot'},

    # "CA": {"xaxis": 't', "yaxis": 'CA'},
    # "CN": {"xaxis": 't', "yaxis": 'CN'},
    })

sim = Sim(t_span=t_span,
          dt=0.01,
          simobjs=[simobj],
          integrate_method="rk4")
# sim.run()
# plotter.instrument_view = True
plotter.animate(sim)
# plotter.plot_obj(simobj)
plotter.show()
# plotter.add_vector()
sim.profiler.print_info()

# with open("temp_data_2.pickle", 'ab') as f:
#     dill.dump(simobj.Y, f)

quit()
X = simobj.Y[ii]
quat = simobj.get_state_array(X, ["q_0", "q_1", "q_2", "q_3"])
yaw_pitch_roll = np.degrees(Rotation.dcm_to_tait_bryan(Rotation.quat_to_dcm(quat)))
divider_str = "-" * 50

print("quat norm:", np.linalg.norm(quat))
print("quat:", quat)
print("tait_bryan:", yaw_pitch_roll)

print(f"\n{divider_str}\nAlpha-Beta\n{divider_str}")
print("alpha:", np.degrees(simobj.get_state_array(X, "alpha")), end=", ")
print("alpha_dot:", np.degrees(simobj.get_state_array(X, "alpha_dot")))
print("beta:", simobj.get_state_array(X, "beta"), end=", ")
print("beta_dot:", simobj.get_state_array(X, "beta_dot"))

print(f"\n{divider_str}\nPQR\n{divider_str}")
print("p:", simobj.get_state_array(X, "p"))
print("q:", simobj.get_state_array(X, "q"))
print("r:", simobj.get_state_array(X, "r"))
print("mass:", simobj.get_state_array(X, "mass"))

print(f"\n{divider_str}\nECI\n{divider_str}")
print("r_i:", simobj.get_state_array(X, ["r_i_x", "r_i_y", "r_i_z"]))
print("v_i:", simobj.get_state_array(X, ["v_i_x", "v_i_y", "v_i_z"]))

print(f"\n{divider_str}\nENU\n{divider_str}")
print("r_enu:", simobj.get_state_array(X, ["r_e", "r_n", "r_u"]))
print("v_enu:", simobj.get_state_array(X, ["v_e", "v_n", "v_u"]))
print("a_enu:", simobj.get_state_array(X, ["a_e", "a_n", "a_u"]))
print("a_norm:", np.linalg.norm(simobj.get_state_array(X, ["a_e", "a_n", "a_u"])))

print(f"\n{divider_str}\nECEF\n{divider_str}")
print("v_e_e:", simobj.get_state_array(X, ["v_e_x", "v_e_y", "v_e_z"]),
      np.linalg.norm(simobj.get_state_array(X, ["v_e_x", "v_e_y", "v_e_z"])))
print("vel_mag_e:", simobj.get_state_array(X, "vel_mag_e"))
print("mach:", simobj.get_state_array(X, "mach"))

print(f"\n{divider_str}\nBODY\n{divider_str}")
print("v_b_e:", simobj.get_state_array(X, ["v_b_e_x", "v_b_e_y", "v_b_e_z"]))
print("v_b_e_hat:", simobj.get_state_array(X, ["v_b_e_hat_x", "v_b_e_hat_y", "v_b_e_hat_z"]),
      np.linalg.norm(simobj.get_state_array(X, ["v_b_e_hat_x", "v_b_e_hat_y", "v_b_e_hat_z"])))
print("g_b_e:", simobj.get_state_array(X, ["g_b_m_x", "g_b_m_y", "g_b_m_z"]),
      np.linalg.norm(simobj.get_state_array(X, ["g_b_m_x", "g_b_m_y", "g_b_m_z"])))
print("a_b_e:", simobj.get_state_array(X, ["a_b_x", "a_b_y", "a_b_z"]),
      np.linalg.norm(simobj.get_state_array(X, ["a_b_x", "a_b_y", "a_b_z"])))

print(f"\n{divider_str}\n{divider_str}")

# dt = 0.01
# vec = "v_e"
# for row in simobj.Y[-10:]:
#     ids = simobj.model.get_state_id([f"{vec}_x", f"{vec}_y", f"{vec}_z"])
#     vmag_id = simobj.model.get_state_id("vel_mag_e")
#     vmag_dot_id = simobj.model.get_state_id("vel_mag_e_dot")
#     v_e_e = row[ids]
#     vel_mag = row[vmag_id]
#     vel_mag_dot = row[vmag_dot_id]
#     print(v_e_e, end=" ")
#     print(np.linalg.norm(v_e_e), end=" ")
#     print(vel_mag, end=" ")
#     # print(vel_mag_dot, end=" ")
#     print()


# ids = simobj.model.get_state_id([f"{vec}_x", f"{vec}_y", f"{vec}_z"])
# cp = simobj.Y[0, ids]
# c = simobj.Y[1, ids]
# delta = np.linalg.norm(c) - np.linalg.norm(cp)
# print(delta / dt)

# print([np.linalg.norm(i) for i in simobj.Y[:10, -5:-2]]) }}}
