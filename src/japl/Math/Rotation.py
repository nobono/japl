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


def quat_to_tait_bryan(q: np.ndarray|quaternion):
    if isinstance(q, quaternion):
        q = q.components #type:ignore
    yaw  = np.arctan2((q[2] * q[3] + q[0] * q[1]), 0.5 - (q[1]**2 + q[2]**2))
    pitch = np.arcsin(-2 * (q[1] * q[3] - q[0] * q[2]))
    roll  = np.arctan2((q[1] * q[2] + q[0] * q[3]), 0.5 - (q[2]**2 + q[3]**2))
    # if abs(2 * (q[1] * q[3] + q[0] * q[2])) == 1.0:
    #     print("gymbal lock conversion")
    return np.array([yaw, pitch, roll])


def tait_bryan_to_dcm(yaw_pitch_roll):
    cyaw = np.cos(yaw_pitch_roll[0])
    cpitch = np.cos(yaw_pitch_roll[1])
    croll = np.cos(yaw_pitch_roll[2])
    syaw = np.sin(yaw_pitch_roll[0])
    spitch = np.sin(yaw_pitch_roll[1])
    sroll = np.sin(yaw_pitch_roll[2])
    dcm = np.array([
        [cpitch*cyaw, sroll*spitch*cyaw - croll*syaw, sroll*syaw + croll*spitch*cyaw],
        [cpitch*syaw, croll*cyaw + sroll*spitch*syaw, croll*spitch*syaw - sroll*cyaw],
        [-spitch, sroll*cpitch, croll*cpitch],
        ])
    return dcm


def dcm_to_tait_bryan(dcm):
    # handle gimbal lock condition
    if abs(abs(dcm[2][0]) - 1) < 1e-12:
        # set roll to zero and solve for yaw
        if dcm[2][0] < 0:
            yaw = np.arctan2(-dcm[0][1], -dcm[0][2])
            pitch = -np.arcsin(max(dcm[2][0], -1.0))
            roll = 0
        else:
            yaw = np.arctan2(dcm[0][1], dcm[0][2])
            pitch = -np.arcsin(min(dcm[2][0], 1.0))
            roll = 0
    else:
        yaw = np.arctan2(dcm[1][0], dcm[0][0])
        pitch = -np.arcsin(dcm[2][0])
        roll = np.arctan2(dcm[2][1], dcm[2][2])
    return np.array([yaw, pitch, roll])


