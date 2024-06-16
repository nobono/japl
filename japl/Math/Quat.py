import numpy as np
from copy import copy
import quaternion
from quaternion import quaternion as Tquaternion



def quat_conj(q):
    assert len(q) == 4
    q[1] *= -1.0
    q[2] *= -1.0
    q[3] *= -1.0
    return q


def quat_array_to_dcm(q: np.ndarray|Tquaternion) -> np.ndarray:
    """This method returns a DCM rotation matrix given a quaternion array"""

    if isinstance(q, Tquaternion):
        return quaternion.as_rotation_matrix(q)

    assert len(q) == 4
    q = copy(q)
    # q = quat_conj(q)
    k = -1.0 # TEMP
    q0_2 = q[0] * q[0]
    q1_2 = q[1] * q[1]
    q2_2 = q[2] * q[2]
    q3_2 = q[3] * q[3]
    q1q2 = q[1] * q[2]
    q0q3 = q[0] * q[3] * k
    q1q3 = q[1] * q[3]
    q0q1 = q[0] * q[1] * k
    q2q3 = q[2] * q[3]
    q0q2 = q[0] * q[2] * k

    dcm = np.zeros(shape=(3,3), dtype=np.float64)
    dcm[0][0] = q0_2 + q1_2 - q2_2 - q3_2
    dcm[0][1] = 2.0 * (q1q2 + q0q3)
    dcm[0][2] = 2.0 * (q1q3 - q0q2)
    dcm[1][0] = 2.0 * (q1q2 - q0q3)
    dcm[1][1] = q0_2 - q1_2 + q2_2 - q3_2
    dcm[1][2] = 2.0 * (q2q3 + q0q1)
    dcm[2][0] = 2.0 * (q1q3 + q0q2)
    dcm[2][1] = 2.0 * (q2q3 - q0q1)
    dcm[2][2] = q0_2 - q1_2 - q2_2 + q3_2

    return dcm




if __name__ == "__main__":
    ang = 0.2
    q1 = quaternion.from_float_array([np.cos(ang/2), np.sin(ang/2), 0, 0])
    q2 = np.array([np.cos(ang/2), np.sin(ang/2), 0 ,0])

    r1 = quaternion.as_rotation_matrix(q1)
    e1 = quaternion.as_euler_angles(q1)

    r2 = quat_array_to_dcm(q2)

