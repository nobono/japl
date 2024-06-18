import numpy as np
import quaternion
from japl.Math.Math import skew
from japl.Math.Quat import quat_array_to_dcm



Quat = quaternion.from_float_array



# e = quaternion.as_euler_angles(q)
# qq = quaternion.from_euler_angles([0, 0, 0])

omega = np.array([
    np.radians(0),
    np.radians(1),
    np.radians(0),
    ])

Skew = skew(omega)

Sw = np.array([
    [0, *-omega],
    [omega[0], *Skew[0]],
    [omega[1], *Skew[1]],
    [omega[2], *Skew[2]],
    ])


q = Quat([1, 0, 0, 0]).components
dt = .01
for i in range(500):
    # q_dot = 0.5 * Sw @ q
    Sq = np.array([
        [-q[1], -q[2], -q[3]],
        [q[0], -q[3], q[2]],
        [q[3], q[0], -q[1]],
        [-q[2], q[1], q[0]],
        ])
    q_dot = 0.5 * Sq @ omega
    q = q_dot * dt + q
    pass
pass
print(q)
ang = quaternion.as_euler_angles(Quat(q))
print(np.degrees(ang))
print(quaternion.from_euler_angles(ang))
