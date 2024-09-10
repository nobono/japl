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
from japl.Library.Vehicles.pog import pog
from japl import Atmosphere
from japl.Util.Results import Results

DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=3)


pog_complete = False
apogee = False

########################################################
# Load
########################################################

atmosphere = Atmosphere()
stage_2_aero = AeroTable(DIR + "/../../../aeromodel/stage_2_aero.mat", from_template="orion")
# aero_thick = AeroTable(DIR + "/../../../aeromodel/stage_1_aero_thick.mat", from_template="orion")
aerotable = AeroTable(DIR + "/../../../aeromodel/stage_1_aero.mat", from_template="orion")

atm_mod = {
        "atmosphere_pressure": atmosphere.pressure,
        "atmosphere_density": atmosphere.density,
        "atmosphere_temperature": atmosphere.temperature,
        "atmosphere_speed_of_sound": atmosphere.speed_of_sound,
        "atmosphere_grav_accel": atmosphere.grav_accel,
        "atmosphere_dynamic_pressure": atmosphere.dynamic_pressure,
        }

model = Model.from_file(DIR + "/mmd.japl", modules=[aerotable.modules, atm_mod])
# model.aerotable = aerotable
# model.aerotable.set_aerotable(DIR + "/../../../aeromodel/stage_1_aero.mat",
#                               from_template="orion")
simobj = SimObject(model)
mass_props = MassPropTable(DIR + "/../../../aeromodel/stage_1_mass.mat", from_template="CMS")

########################################################
# Custom Input Function
########################################################


def ned_to_body_cmd(t: float, quat: np.ndarray, vec: np.ndarray):
    omega_e = Earth.omega
    q_0, q_1, q_2, q_3 = quat
    r_e_m = X[27:30]
    C_eci_to_ecef = np.array([
        [np.cos(omega_e * t), np.sin(omega_e * t), 0],
        [-np.sin(omega_e * t), np.cos(omega_e * t), 0],
        [0, 0, 1]])
    C_body_to_eci = np.array([
        [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],  # type:ignore # noqa
        [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],  # type:ignore # noqa
        [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa
    C_body_to_ecef = C_eci_to_ecef @ C_body_to_eci
    lla0 = Rotation.ecef_to_lla(r_e_m)
    lat0, lon0, _ = lla0
    C_ecef_to_enu = np.array([
        [-np.sin(lon0), np.cos(lon0), 0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ])
    acc_cmd_ecef = C_ecef_to_enu.T @ np.asarray(vec)
    acc_cmd_body = C_body_to_ecef.T @ acc_cmd_ecef
    acc_cmd_body = 60 * (acc_cmd_body / np.linalg.norm(acc_cmd_body))
    a_c_y = acc_cmd_body[1]
    a_c_z = acc_cmd_body[2]
    return (a_c_y, a_c_z)


def user_input_func(t, X, U, S, dt, simobj: SimObject):
    global mass_props
    global pog_complete
    global apogee

    do_pog = True
    do_ld_guidance = False

    # 1.9665 drop 1 stage
    # 2.4665 2nd stage start

    alpha = simobj.get_state_array(X, "alpha")
    r_enu_m = simobj.get_state_array(X, ["r_e", "r_n", "r_u"])
    v_enu_m = simobj.get_state_array(X, ["v_e", "v_n", "v_u"])
    alt = r_enu_m[2]
    body_rates = simobj.get_state_array(X, ["p", "q", "r"])
    mass = simobj.get_state_array(X, "wet_mass")
    q_bar = simobj.get_state_array(X, "q_bar")
    mach = simobj.get_state_array(X, "mach")

    a_c_y = 0.00001
    a_c_z = 0.00001
    if do_pog:
        if t > 2.0:
            S[-1] = 0
        else:
            if not pog_complete or t < 0.1:
                pog_complete, a_c = pog(t,
                                        desired_vleg=np.radians(45),
                                        desired_bearing_angle=0,
                                        alphaTotal=alpha,
                                        altm=alt,
                                        lead_angle=0,
                                        vm=v_enu_m,
                                        body_rates=body_rates)
                a_c_y = a_c[1]
                a_c_z = a_c[2]
                if pog_complete:
                    S[-1] = 0
                    print("POG pog_complete at t=%.2f" % t)
    if do_ld_guidance:
        if not apogee:
            apogee = v_enu_m[2] < 0.0
        else:
            # print("apogee reached")
            # do L/D guidance
            Sref = aerotable.get_Sref()
            opt_CL, opt_CD, opt_alpha = aerotable.ld_guidance(alpha=alpha, mach=mach, alt=alt)  # type:ignore
            f_l = opt_CL * q_bar * Sref
            f_d = opt_CD * q_bar * Sref
            a_l = f_l / mass
            a_d = f_d / mass
            a_c_y = 0
            a_c_z = -a_l
            # print(a_c_y, a_c_z)

    # 2nd stage stuff
    # if t >= 1.9665 and t < 2.5664:
    #     a_c_y = 0.00001
    #     a_c_z = 0.00001
    # if t >= 2.5665:
    #     aerotable.set(stage_2_aero)
    #     simobj.set_state_array(X, "wet_mass", 11.848928)

    pressure = atmosphere.pressure(alt)
    U[2] = a_c_y                                # acc cmd
    U[3] = a_c_z                                # acc cmd
    U[4] = mass_props.get_thrust(t, pressure)   # thrust
    U[5] = mass_props.get_mass_dot(t)           # mass_dot


########################################################
# Add user-function
########################################################

model.set_input_function(user_input_func)

########################################################
# Setup plotter
########################################################

plotter = PyQtGraphPlotter(frame_rate=30,
                           figsize=[10, 10],
                           aspect="auto",
                           ff=5.0,
                           axis_color="grey",
                           )

########################################################
# Initialize Model
########################################################

t_span = [0, 200]
ecef0 = [Earth.radius_equatorial, 0, 0]

r0_enu = [0, 0, 0]
v0_enu = [0, 0, 1]
a0_enu = [0, 0, -9.81]
# dcm = Rotation.tait_bryan_to_dcm(np.radians([0, 0, 0])).T  # yaw-pitch-roll
# quat0 = quaternion.from_rotation_matrix(dcm).components
# quat0 = [1, 0, 0, 0]
quat0 = quaternion.from_euler_angles(np.radians([0, -45, 0])).components

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
thrust0 = 0

v0_ecef = v0_ecef

v0_body = C_body_to_eci.T @ v0_eci
v0_body_hat = v0_body / np.linalg.norm(v0_body)
g0_body = [0, 0, -9.81]
a0_body = [0, 0, -9.81]

wet_mass0 = 108 / 2.2  # mass_props.wet_mass  # + (24.1224 / 2.2)
dry_mass0 = mass_props.dry_mass
# print(wet_mass0)

lift0 = 0
drag0 = 0

CA0 = 0
CNB0 = 0
q_bar0 = 0

simobj.init_state([quat0,
                   r0_eci, v0_eci,
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
                   drag0,
                   ])

omega_n = 20  # natural frequency
zeta = 0.7    # damping ratio
K_phi = 1     # roll gain
omega_p = 20  # natural frequency (roll)
phi_c = 0     # roll angle command
T_r = 0.5     # roll autopilot time constant
flag_boosting = 1
stage = 1

simobj.init_static([
    omega_n,
    zeta,
    K_phi,
    omega_p,
    phi_c,
    T_r,
    flag_boosting,
    stage,
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

    "N-U": {"xaxis": 'r_n', "yaxis": 'r_u',
            # "xlim": [0, 7e3],
            # "ylim": [0, 7e3]
            },
    # "N-E": {"xaxis": 'r_n', "yaxis": 'r_e',
    #         "xlim": [0, 7e3],
    #         "ylim": [0, 7e3]
    #         },

    # "v_e": {"xaxis": 'r_n', "yaxis": 'v_e'},
    # "v_n": {"xaxis": 'r_n', "yaxis": 'v_n'},
    # "v_u": {"xaxis": 'r_n', "yaxis": 'v_u'},

    # "a_u": {"xaxis": 'r_n', "yaxis": 'a_u'},

    # "Mach": {"xaxis": 't', "yaxis": 'mach'},
    # "Thrust": {"xaxis": 't', "yaxis": 'thrust'},
    # "Drag": {"xaxis": 't', "yaxis": 'drag'},
    # "V": {"xaxis": 't', "yaxis": 'vel_mag_e'},

    # "alpha": {"xaxis": 't', "yaxis": 'alpha'},
    # "beta": {"xaxis": 't', "yaxis": 'beta'},
    # "phi_hat": {"xaxis": 't', "yaxis": 'phi_hat'},
    "alpha_c": {"xaxis": 't', "yaxis": 'alpha_c'},

    # "p": {"xaxis": 't', "yaxis": 'p'},
    # "q": {"xaxis": 't', "yaxis": 'q'},
    # "r": {"xaxis": 't', "yaxis": 'r'},

    "wet_mass": {"xaxis": 't', "yaxis": 'wet_mass'},
    # "dry_mass": {"xaxis": 't', "yaxis": 'dry_mass'},
    # "mass_dot": {"xaxis": 't', "yaxis": 'mass_dot'},

    # "CA": {"xaxis": 't', "yaxis": 'CA'},
    # "CN": {"xaxis": 't', "yaxis": 'CN'},

    # "lift": {"xaxis": 't', "yaxis": 'lift'},
    # "drag": {"xaxis": 't', "yaxis": 'drag'},

    # "a_c_y": {"xaxis": 't', "yaxis": 'a_c_y'},
    # "a_c_z": {"xaxis": 't', "yaxis": 'a_c_z'},
    })


########################################################
# Events
########################################################

def event_hit_ground(t, X, U, S, dt, simobj) -> bool:
    if simobj.get_state_array(X, "r_u") < 0:
        return True
    else:
        return False

########################################################
# Sim
########################################################


sim = Sim(t_span=t_span,
          dt=0.01,
          simobjs=[simobj],
          integrate_method="rk4")

sim.add_event(event_hit_ground, "stop")
# sim.run()
# plotter.instrument_view = True
plotter.animate(sim).show()
# plotter.plot_obj(simobj).show()
# plotter.add_vector()
# plotter.show()
# sim.profiler.print_info()

# create dict of outputs



out = Results(sim.T, simobj)
out.save("./run1_ballistic.pickle")

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
