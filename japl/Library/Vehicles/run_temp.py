import os
import argparse
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
AEROMODEL_DIR = DIR + "/../../../aeromodel/"
DATA_DIR = DIR + "/../../../data/"
np.set_printoptions(suppress=True, precision=3)


pog_complete = False
apogee = False
stage_sep = False

########################################################
# Load
########################################################
stage_2_aero = AeroTable(AEROMODEL_DIR + "stage_2_aero.mat", from_template="orion")
stage_1_aero = AeroTable(AEROMODEL_DIR + "stage_1_aero.mat", from_template="orion")
# stage_1_aero_thick = AeroTable(AEROMODEL_DIR + "stage_1_aero_thick.mat", from_template="orion")
stage_1_mass = MassPropTable(AEROMODEL_DIR + "stage_1_mass.mat", from_template="CMS")

atmosphere = Atmosphere()
aerotable = AeroTable()
mass_props = MassPropTable()

aerotable.set(stage_1_aero)
aerotable.add_stage(stage_1_aero)
aerotable.add_stage(stage_2_aero)
mass_props.set(stage_1_mass)

model = Model.from_file(DATA_DIR + "mmd.japl", modules=[aerotable.modules, atmosphere.modules])
simobj = SimObject(model)

########################################################
# Custom Input Function
########################################################


def ned_to_body_cmd(t: float, quat: np.ndarray, r_ecef: np.ndarray, vec: np.ndarray):
    omega_e = Earth.omega
    q_0, q_1, q_2, q_3 = quat
    C_eci_to_ecef = np.array([
        [np.cos(omega_e * t), np.sin(omega_e * t), 0],
        [-np.sin(omega_e * t), np.cos(omega_e * t), 0],
        [0, 0, 1]])
    C_body_to_eci = np.array([
        [1 - 2*(q_2**2 + q_3**2), 2*(q_1*q_2 + q_0*q_3) , 2*(q_1*q_3 - q_0*q_2)],  # type:ignore # noqa
        [2*(q_1*q_2 - q_0*q_3) , 1 - 2*(q_1**2 + q_3**2), 2*(q_2*q_3 + q_0*q_1)],  # type:ignore # noqa
        [2*(q_1*q_3 + q_0*q_2) , 2*(q_2*q_3 - q_0*q_1), 1 - 2*(q_1**2 + q_2**2)]]).T  # type:ignore # noqa
    C_body_to_ecef = C_eci_to_ecef @ C_body_to_eci
    lla0 = Rotation.ecef_to_lla(r_ecef)
    lat0, lon0, _ = lla0
    C_ecef_to_enu = np.array([
        [-np.sin(lon0), np.cos(lon0), 0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ])
    vec_ecef = C_ecef_to_enu.T @ np.asarray(vec)
    vec_body = C_body_to_ecef.T @ vec_ecef
    vec_body = 60 * (vec_body / np.linalg.norm(vec_body))
    a_c_y = vec_body[1]
    a_c_z = vec_body[2]
    return (a_c_y, a_c_z)


def user_input_func(t, X, U, S, dt, simobj: SimObject):
    global mass_props
    global pog_complete
    global apogee
    global stage_sep
    global aerotable

    # print(t)

    do_pog = True
    do_ld_guidance = True

    alpha = simobj.get_state_array(X, "alpha")
    r_enu_m = simobj.get_state_array(X, ["r_e", "r_n", "r_u"])
    v_enu_m = simobj.get_state_array(X, ["v_e", "v_n", "v_u"])
    alt = r_enu_m[2]
    vel_up = v_enu_m[2]
    body_rates = simobj.get_state_array(X, ["p", "q", "r"])
    mass = simobj.get_state_array(X, "wet_mass")
    q_bar = simobj.get_state_array(X, "q_bar")
    mach = simobj.get_state_array(X, "mach")
    boosting = simobj.get_static_array(S, "flag_boosting")
    stage = simobj.get_static_array(S, "stage")

    if t % 5 == 0:
        print(f"t:%.2f, range:%.2f, mach:%.2f, alt:%.2f" % (
            t, r_enu_m[1], mach, alt
            ))

    a_c_y = 0.00001
    a_c_z = 0.00001
    if do_pog and not pog_complete:
        if not pog_complete or t < 0.1:
            pog_complete, a_c = pog(t,
                                    desired_vleg=np.radians(67),
                                    desired_bearing_angle=0,
                                    alphaTotal=alpha,
                                    altm=alt,
                                    lead_angle=0,
                                    vm=v_enu_m,
                                    body_rates=body_rates)
            a_c_y = a_c[1]
            a_c_z = a_c[2]
            if pog_complete:
                print("POG pog_complete at t=%.2f (s)" % t)

    if do_ld_guidance:
        if not apogee:
            if(apogee := vel_up < 0.0):
                print("Apogee reached %.2f @ t=%.2f (s)" % (alt, t))
        else:
            # do L/D guidance
            Sref = aerotable.get_Sref()
            opt_CL, opt_CD, opt_alpha = aerotable.ld_guidance(stage=stage,  # type:ignore
                                                              boosting=boosting,  # type:ignore
                                                              alpha=alpha,  # type:ignore
                                                              mach=mach, # type:ignore
                                                              alt=alt)  # type:ignore
            f_l = opt_CL * q_bar * Sref
            f_d = opt_CD * q_bar * Sref
            a_l = f_l / mass
            a_d = f_d / mass
            a_c_y = 0
            a_c_z = -a_l

    t_coast_phase = [1.9665, 2.4664]
    t_stage_1_sep = 2.4665
    # 2nd stage stuff
    if t >= t_coast_phase[0] and t < t_coast_phase[1]:
        a_c_y = 0.00001
        a_c_z = 0.00001
    if t >= t_stage_1_sep and not stage_sep:
        print("stage sep @ t=%.2f (s)" % t)
        # aerotable.set(stage_2_aero)
        simobj.set_static_array(S, "stage", 2)
        simobj.set_state_array(X, "wet_mass", 11.848928)
        stage_sep = True

    pressure = atmosphere.pressure(alt)
    thrust = mass_props.get_thrust(t, pressure)
    simobj.set_static_array(S, "flag_boosting", (thrust > 0.0))

    U[2] = a_c_y                                # acc cmd
    U[3] = a_c_z                                # acc cmd
    U[4] = thrust                               # thrust
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
                           ff=1.0,
                           axis_color="grey",
                           )

########################################################
# Initialize Model
########################################################

ecef0 = [Earth.radius_equatorial, 0, 0]

r0_enu = [0, 0, 0]
v0_enu = [0, 0, 1]
a0_enu = [0, 0, -9.81]
# dcm = Rotation.tait_bryan_to_dcm(np.radians([0, 0, 0])).T  # yaw-pitch-roll
# quat0 = quaternion.from_rotation_matrix(dcm).components
# quat0 = [1, 0, 0, 0]
# quat0 = quaternion.from_euler_angles(np.radians([0, -70, 0])).components
ang = 67
ang = np.radians(ang - 90.0)
quat0 = [np.cos(ang / 2), 0, np.sin(ang / 2), 0]

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

# TODO make set_config() a method
# which appends accaptable arguments / dict
simobj.plot.set_config({

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

    "Mach": {"xaxis": 't', "yaxis": 'mach'},
    # "Thrust": {"xaxis": 't', "yaxis": 'thrust'},
    # "Drag": {"xaxis": 't', "yaxis": 'drag'},
    # "V": {"xaxis": 't', "yaxis": 'vel_mag_e'},

    "alpha": {"xaxis": 't', "yaxis": 'alpha'},
    # "beta": {"xaxis": 't', "yaxis": 'beta'},
    # "phi_hat": {"xaxis": 't', "yaxis": 'phi_hat'},
    # "alpha_c": {"xaxis": 't', "yaxis": 'alpha_c'},

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script using argparse")
    parser.add_argument('--show',
                        dest="show",
                        default=False,
                        action="store_true",
                        help='displays the plots')
    parser.add_argument('-t',
                        type=int,
                        dest="t",
                        default=400,
                        help='select sim end time in seconds')
    parser.add_argument('-o',
                        dest="output",
                        type=str,
                        default="",
                        help='Output file name')
    args = parser.parse_args()

    sim = Sim(t_span=[0, args.t],
              dt=0.01,
              simobjs=[simobj],
              integrate_method="rk4")

    sim.add_event(event_hit_ground, "stop")

    if args.show:
        plotter.animate(sim).show()
    else:
        sim.run()
        sim.profiler.print_info()

    # plotter.plot_obj(simobj).show()

    out = Results(sim.T, simobj)
    if args.output:
        out.save(DIR + f"/../../../data/{args.output}.pickle")
