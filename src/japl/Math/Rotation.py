import numpy as np
from quaternion.numpy_quaternion import quaternion



def quat_conj(q):
    assert len(q) == 4
    q[1] *= -1.0
    q[2] *= -1.0
    q[3] *= -1.0
    return q


def quat_to_dcm(q: np.ndarray|quaternion) -> np.ndarray:
    """This method returns a DCM rotation matrix given a quaternion array"""

    if isinstance(q, quaternion):
        q = q.components.copy() #type:ignore
    else:
        q = q.copy()

    k = -1.0    # (1 or -1) conjugates input quaternion depending on convention
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


