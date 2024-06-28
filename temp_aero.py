import numpy as np

import matplotlib.pyplot as plt
import plotext as plx

from japl import Atmosphere
from japl import SimObject
from japl import Model
from japl import AeroTable


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
simobj = vehicle
assert simobj.aerotable

fig, ax = plt.subplots(2, 2, figsize=(12, 10))


######################3
alpha = 0
phi = 0
mach = 2.0
iota = 0
alt = 10_000

for iota in np.linspace(0, np.radians(40), 1):

    alphas = []
    iotas = []
    CNBs = []
    Mys = []
    forces = []

    for alpha in np.linspace(-np.radians(40), np.radians(40), 100):
        # alpha *= -1

        vel = mach * atmosphere.speed_of_sound(alt)

        CLMB = simobj.aerotable.get_CLMB_Total(alpha, phi, mach, iota)
        CNB = simobj.aerotable.get_CNB_Total(alpha, phi, mach, iota)

        My_coef = CLMB + (simobj.cg - simobj.aerotable.MRC[0]) * CNB

        q = atmosphere.dynamic_pressure(vel, alt)
        My = My_coef * q * simobj.aerotable.Sref * simobj.aerotable.Lref
        zforce = CNB * q * simobj.aerotable.Sref

        alphas += [alpha]
        iotas += [iota]
        CNBs += [CNB]
        Mys += [My]
        forces += [zforce]

    ax[0, 0].plot(alphas, Mys)
    ax[0, 0].set_xlabel("alpha")
    ax[0, 0].set_ylabel("Mys")
    ax[0, 0].grid()

    ax[0, 1].plot(alphas, forces)
    ax[0, 1].set_xlabel("alpha")
    ax[0, 1].set_ylabel("force")
    ax[0, 1].grid()

######################3

alpha = 0
phi = 0
mach = 2.0
iota = 0
alt = 10_000

alphas = []
iotas = []
CNBs = []
Mys = []
forces = []

for iota in np.linspace(-np.radians(40), np.radians(40), 100):

    vel = mach * atmosphere.speed_of_sound(alt)

    CLMB = simobj.aerotable.get_CLMB_Total(alpha, phi, mach, iota)
    CNB = simobj.aerotable.get_CNB_Total(alpha, phi, mach, iota)

    My_coef = CLMB + (simobj.cg - simobj.aerotable.MRC[0]) * CNB

    q = atmosphere.dynamic_pressure(vel, alt)
    My = My_coef * q * simobj.aerotable.Sref * simobj.aerotable.Lref
    zforce = CNB * q * simobj.aerotable.Sref

    alphas += [alpha]
    iotas += [iota]
    CNBs += [CNB]
    Mys += [My]
    forces += [zforce]

ax[1, 0].plot(iotas, Mys)
ax[1, 0].set_xlabel("iota")
ax[1, 0].set_ylabel("Mys")
ax[1, 0].grid()

ax[1, 1].plot(iotas, forces)
ax[1, 1].set_xlabel("iota")
ax[1, 1].set_ylabel("force")
ax[1, 1].grid()


# plx.from_matplotlib(fig)
# plx.show()
plt.show()
quit()

