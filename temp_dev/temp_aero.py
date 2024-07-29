import numpy as np

import matplotlib.pyplot as plt
import plotext as plx

from japl import Atmosphere
from japl import SimObject
from japl import Model
from japl import AeroTable
from japl.Math.Rotation import quat_to_tait_bryan
from japl.Math.Vec import vec_ang


# ---------------------------------------------------

# Model
####################################
model = Model()

x  = model.add_state("x",         0,  "x (m)")
y  = model.add_state("y",         1,  "y (m)")
z  = model.add_state("z",         2,  "z (m)")
vx = model.add_state("vx",        3,  "xvel (m/s)")
vy = model.add_state("vy",        4,  "yvel (m/s)")
vz = model.add_state("vz",        5,  "zvel (m/s)")
wx = model.add_state("wx",        6,  "wx (rad/s)")
wy = model.add_state("wy",        7,  "wy (rad/s)")
wz = model.add_state("wz",        8,  "wz (rad/s)")
q0 = model.add_state("q0",        9,  "q0")
q1 = model.add_state("q1",        10, "q1")
q2 = model.add_state("q2",        11, "q2")
q3 = model.add_state("q3",        12, "q3")

mass = model.add_state("mass",    13, "mass (kg)")


Sq = np.array([
    [-q1, -q2, -q3],
    [q0, -q3, q2],
    [q3, q0, -q1],
    [-q2, q1, q0],
    ]) * 0.5

A = np.array([
    [0,0,0,  1,0,0,  0,0,0,  0,0,0,0,  0], # x
    [0,0,0,  0,1,0,  0,0,0,  0,0,0,0,  0], # y
    [0,0,0,  0,0,1,  0,0,0,  0,0,0,0,  0], # z
    [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # vx
    [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # vy
    [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # vz

    [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # wx
    [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # wy
    [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # wz

    [0,0,0,  0,0,0,  *Sq[0], 0,0,0,0,  0], # q0
    [0,0,0,  0,0,0,  *Sq[1], 0,0,0,0,  0], # q1
    [0,0,0,  0,0,0,  *Sq[2], 0,0,0,0,  0], # q2
    [0,0,0,  0,0,0,  *Sq[3], 0,0,0,0,  0], # q3

    [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0], # mass
    ])

B = np.array([
    # force  torque
    [0,0,0,  0,0,0],
    [0,0,0,  0,0,0],
    [0,0,0,  0,0,0],
    [1,0,0,  0,0,0],
    [0,1,0,  0,0,0],
    [0,0,1,  0,0,0],

    [0,0,0,  1,0,0],
    [0,0,0,  0,1,0],
    [0,0,0,  0,0,1],

    [0,0,0,  0,0,0],
    [0,0,0,  0,0,0],
    [0,0,0,  0,0,0],
    [0,0,0,  0,0,0],

    [0,0,0,  0,0,0],
    ])

model.ss(A, B)

vehicle = SimObject(model=model, size=2, color='tab:blue')
vehicle.aerotable = AeroTable("./aeromodel/aeromodel_psb.mat")
atmosphere = Atmosphere()


def aero_update(simobj: SimObject, alpha, phi, mach, iota, alt):
    assert simobj.aerotable
    vel = mach * atmosphere.speed_of_sound(alt)

    CLMB = -simobj.aerotable.get_CLMB_Total(alpha, phi, mach, iota)
    CNB = simobj.aerotable.get_CNB_Total(alpha, phi, mach, iota)

    My_coef = CLMB + (simobj.cg - simobj.aerotable.MRC[0]) * CNB

    q = atmosphere.dynamic_pressure(vel, alt)
    My = My_coef * q * simobj.aerotable.Sref * simobj.aerotable.Lref
    zforce = CNB * q * simobj.aerotable.Sref

    return (alpha, iota, CNB, My, zforce)


def reset():
    alpha = 0
    phi = 0
    mach = 0
    iota = 0
    alt = 10_000

    alphas = []
    iotas = []
    CNBs = []
    Mys = []
    forces = []

    return (alpha, phi, mach, iota, alt,
            alphas, iotas, CNBs, Mys, forces)


######################

alpha, phi, mach, iota, alt,\
        alphas, iotas, CNBs, Mys, forces = reset()

ang = np.radians(0)
quat0 = [np.cos(ang/2), 0, np.sin(ang/2), 0]
vehicle.init_state([0,0,alt, 700,0,0, 0,0,0, quat0, 0])
vehicle._pre_sim_checks()
simobj = vehicle
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

dt = 0.01
X = vehicle.X0.copy()

ts = []
poss = []
vels = []
angvels = []
quats = []
pitchs = []
fpa = []

for i in range(30):

    alt = simobj.get_state(X, "z")
    vel = simobj.get_state(X, ["vx", "vy", "vz"])
    quat = simobj.get_state(X, ["q1", "q1", "q2", "q3"])

    # calculate current mach
    speed = float(np.linalg.norm(vel))
    mach = (speed /atmosphere.speed_of_sound(alt)) #type:ignore

    # calc angle of attack: (pitch_angle - flight_path_angle)
    vel_hat = vel / speed                                       # flight path vector

    # projection vel_hat --> x-axis
    zx_plane_norm = np.array([0, 1, 0])
    vel_hat_zx = ((vel_hat @ zx_plane_norm) / np.linalg.norm(zx_plane_norm)) * zx_plane_norm
    vel_hat_proj = vel_hat - vel_hat_zx

    # get Trait-bryan angles (yaw, pitch, roll)
    yaw_angle, pitch_angle, roll_angle = quat_to_tait_bryan(np.asarray(quat))

    # angle between proj vel_hat & xaxis
    x_axis_intertial = np.array([1, 0, 0])
    flight_path_angle = np.sign(vel_hat_proj[2]) * vec_ang(vel_hat_proj, x_axis_intertial)
    alpha = pitch_angle - flight_path_angle                     # angle of attack
    phi = roll_angle

    iota = np.radians(5)
    alpha, iota, CNB, My, zforce = aero_update(vehicle, alpha, phi, mach, iota, alt)

    acc = zforce / simobj.mass
    ang_acc = My / simobj.Iyy


    U = np.array([0,0,acc, 0,ang_acc,0])
    Xdot = vehicle.model.step(X, U)
    X = Xdot * dt + X

    ts += [i * dt]
    poss += [X[:3]]
    vels += [X[3:6]]
    angvels += [X[6:9]]
    quats += [X[9:13]]
    pitchs += [np.degrees(pitch_angle)]
    alphas += [np.degrees(alpha)]
    CNBs += [CNB]
    Mys += [My]
    forces += [zforce]
    fpa += [np.degrees(flight_path_angle)]

poss = np.asarray(poss)
vels = np.asarray(vels)
angvels = np.asarray(angvels)
quats = np.asarray(quats)


ax[0, 0].plot(ts, alphas)
ax[0, 0].set_xlabel("t")
ax[0, 0].set_ylabel("alpha")
ax[0, 0].grid()

ax[0, 1].plot(ts, poss[:, 2])
ax[0, 1].set_xlabel("t")
ax[0, 1].set_ylabel("alt")
ax[0, 1].grid()

ax[1, 0].plot(ts, Mys)
ax[1, 0].set_xlabel("t")
ax[1, 0].set_ylabel("torque")
ax[1, 0].grid()

ax[1, 1].plot(ts, forces)
ax[1, 1].set_xlabel("t")
ax[1, 1].set_ylabel("force")
ax[1, 1].grid()

# for iota in np.linspace(0, np.radians(40), 1):

#     alpha, phi, mach, iota, alt,\
#             alphas, iotas, CNBs, Mys, forces = reset()

#     for alpha in np.linspace(0, np.radians(40), 100):
#         alpha, iota, CNB, My, zforce = aero_update(vehicle, alpha, phi, mach, iota, alt)

#         alphas += [alpha]
#         iotas += [iota]
#         CNBs += [CNB]
#         Mys += [My]
#         forces += [zforce]

#     ax[0, 0].plot(alphas, Mys)
#     ax[0, 0].set_xlabel("alpha")
#     ax[0, 0].set_ylabel("Mys")
#     ax[0, 0].grid()

#     ax[0, 1].plot(alphas, forces)
#     ax[0, 1].set_xlabel("alpha")
#     ax[0, 1].set_ylabel("force")
#     ax[0, 1].grid()


# ######################


# alpha, phi, mach, iota, alt,\
#         alphas, iotas, CNBs, Mys, forces = reset()

# for iota in np.linspace(0, np.radians(40), 100):

#     alpha, iota, CNB, My, zforce = aero_update(vehicle, alpha, phi, mach, iota, alt)

#     alphas += [alpha]
#     iotas += [iota]
#     CNBs += [CNB]
#     Mys += [My]
#     forces += [zforce]

# ax[1, 0].plot(iotas, Mys)
# ax[1, 0].set_xlabel("iota")
# ax[1, 0].set_ylabel("Mys")
# ax[1, 0].grid()

# ax[1, 1].plot(iotas, forces)
# ax[1, 1].set_xlabel("iota")
# ax[1, 1].set_ylabel("force")
# ax[1, 1].grid()


plx.from_matplotlib(fig)
plx.show()
# plt.show()
quit()

