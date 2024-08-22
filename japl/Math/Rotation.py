import numpy as np
from quaternion.numpy_quaternion import quaternion



def Sq(q: np.ndarray, dtype: type = float) -> np.ndarray:
    """quaternion dynamics matrix q_dot = dt * 0.5 * Sq * ang_vel"""
    assert len(q) == 4
    q0, q1, q2, q3 = q
    return np.array([[-q1, -q2, -q3],
                     [q0, -q3, q2],
                     [q3, q0, -q1],
                     [-q2, q1, q0]], dtype=dtype)


def Sw(ang_vel: np.ndarray, dtype: type = float) -> np.ndarray:
    """quaternion dynamics matrix q_dot = -dt * 0.5 * Sw * q"""
    assert len(ang_vel) == 3
    wx, wy, wz = ang_vel
    return np.array([[0, -wx, -wy, -wz],
                     [wx, 0, wz, -wy],
                     [wy, -wz, 0, wx],
                     [wz, wy, -wx, 0]], dtype=dtype)


def quat_conj(q: np.ndarray, dtype: type = float) -> np.ndarray:
    assert len(q) == 4
    ret = np.array([q[0], -q[1], -q[2], -q[3]], dtype=dtype)
    return ret


def quat_norm(q: np.ndarray) -> np.ndarray:
    assert len(q) == 4
    return (q / np.linalg.norm(q))


def quat_mult(q: np.ndarray, p: np.ndarray, dtype: type = float) -> np.ndarray:
    return np.array([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                     p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                     p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                     p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]], dtype=dtype)


def quat_to_dcm(q: np.ndarray|quaternion, dtype: type = float) -> np.ndarray:
    """This method returns a DCM rotation matrix given a quaternion array"""
    if isinstance(q, quaternion):
        q = q.components.copy()  # type:ignore
    else:
        q = q.copy()

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    # This form removes q1 from the 0,0, q2 from the 1,1 and q3 from the 2,2 entry and results
    # in a covariance prediction that is better conditioned.
    # It requires the quaternion to be unit length and is mathematically identical
    # to the alternate form when q0**2 + q1**2 + q2**2 + q3**2 = 1
    # See https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    dcm = np.array([[1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3)    , 2*(q1*q3 + q0*q2)    ],   # type:ignore # noqa
                    [2*(q1*q2 + q0*q3)     , 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)    ],   # type:ignore # noqa
                    [2*(q1*q3-q0*q2)       , 2*(q2*q3 + q0*q1)    , 1 - 2*(q1**2 + q2**2)]])  # type:ignore # noqa
    return dcm


def quat_to_tait_bryan(q: np.ndarray|quaternion, dtype: type = float) -> np.ndarray:
    if isinstance(q, quaternion):
        q = q.components  # type:ignore
    dcm = quat_to_dcm(q, dtype=dtype)
    return dcm_to_tait_bryan(dcm, dtype=dtype)


def tait_bryan_to_dcm(yaw_pitch_roll, dtype: type = float):
    cyaw = np.cos(yaw_pitch_roll[0], dtype=dtype)
    cpitch = np.cos(yaw_pitch_roll[1], dtype=dtype)
    croll = np.cos(yaw_pitch_roll[2], dtype=dtype)
    syaw = np.sin(yaw_pitch_roll[0], dtype=dtype)
    spitch = np.sin(yaw_pitch_roll[1], dtype=dtype)
    sroll = np.sin(yaw_pitch_roll[2], dtype=dtype)
    dcm = np.array([
        [cpitch * cyaw, sroll * spitch * cyaw - croll * syaw, sroll * syaw + croll * spitch * cyaw],
        [cpitch * syaw, croll * cyaw + sroll * spitch * syaw, croll * spitch * syaw - sroll * cyaw],
        [-spitch, sroll * cpitch, croll * cpitch],
        ], dtype=dtype)
    return dcm


def dcm_to_tait_bryan(dcm: np.ndarray, dtype: type = float) -> np.ndarray:
    # handle gimbal lock condition
    if abs(abs(dcm[2][0]) - 1) < 1e-15:
        # set roll to zero and solve for yaw
        if dcm[2][0] < 0:
            yaw = np.arctan2(-dcm[0][1], -dcm[0][2], dtype=dtype)
            pitch = -np.arcsin(max(dcm[2][0], -1.0), dtype=dtype)
            roll = 0
        else:
            yaw = np.arctan2(dcm[0][1], dcm[0][2], dtype=dtype)
            pitch = -np.arcsin(min(dcm[2][0], 1.0), dtype=dtype)
            roll = 0
    else:
        yaw = np.arctan2(dcm[1][0], dcm[0][0], dtype=dtype)
        pitch = -np.arcsin(dcm[2][0], dtype=dtype)
        roll = np.arctan2(dcm[2][1], dcm[2][2], dtype=dtype)
    return np.array([yaw, pitch, roll], dtype=dtype)
