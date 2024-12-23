from typing import Optional
import numpy as np
from japl.Library.Earth.Earth import Earth
from quaternion.numpy_quaternion import quaternion
sin = np.sin
cos = np.cos


def euler_to_dcm(roll: float = 0, pitch: float = 0, yaw: float = 0) -> np.ndarray:
    R = np.array([
        [cos(yaw)*cos(pitch), cos(yaw)*sin(pitch*sin(roll))-sin(yaw)*cos(roll), cos(yaw)*sin(pitch)*cos(roll)+sin(yaw)*sin(roll)],  # type:ignore # noqa
        [sin(yaw)*cos(pitch), sin(yaw)*sin(pitch)*sin(roll)+cos(yaw)*cos(roll), sin(yaw)*sin(pitch)*cos(roll)-cos(yaw)*sin(roll)],  # type:ignore # noqa
        [-sin(pitch), cos(pitch)*sin(roll), cos(pitch)*cos(roll)]  # type:ignore # noqa
        ])
    return R


def body_to_enu(t: float, body_xyz: np.ndarray, quat: np.ndarray, r_ecef: np.ndarray):
    """This method converts a vector from Body-frame
    coordinates to ENU coordinates.

    ---------------------------------------------------
    Arguments:
        - t: time in seconds
        - body_xyz: [x, y, z] body-frame vector
        - quat: quaternion (body to eci)
        - r_ecef: [x, y, z] ecef position coordinates

    Returns:
        - enu: [east, north, up] ENU-coodinates
    ---------------------------------------------------
    """
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
    lla0 = ecef_to_lla(r_ecef)
    lat0, lon0, _ = lla0
    C_ecef_to_enu = np.array([
        [-np.sin(lon0), np.cos(lon0), 0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ])
    vec_ecef = C_body_to_ecef @ body_xyz
    enu = C_ecef_to_enu @ vec_ecef
    return enu


def enu_to_body(t: float, quat: np.ndarray, r_ecef: np.ndarray, enu: np.ndarray):
    """This method converts a position vector from ENU
    coordinates to Body-frame coordinates.

    ---------------------------------------------------
    Arguments:
        - t: time in seconds
        - quat: quaternion (body to eci)
        - r_ecef: [x, y, z] ecef position coordinates
        - enu: [east, north, up] ENU-coodinates

    Returns:
        - body_xyz: [x, y, z] BODY-coordinates
    ---------------------------------------------------
    """
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
    lla0 = ecef_to_lla(r_ecef)
    lat0, lon0, _ = lla0
    C_ecef_to_enu = np.array([
        [-np.sin(lon0), np.cos(lon0), 0],
        [-np.sin(lat0) * np.cos(lon0), -np.sin(lat0) * np.sin(lon0), np.cos(lat0)],
        [np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)],
        ])
    vec_ecef = C_ecef_to_enu.T @ np.asarray(enu)
    body_xyz = C_body_to_ecef.T @ vec_ecef
    return body_xyz


def eci_to_ecef(eci_xyz: np.ndarray|list, t: float = 0) -> np.ndarray:
    """This method converts a position vector from ECI
    coordinates to ECEF.

    ---------------------------------------------------
    Arguments:
        - eci_xyz: [x, y, z] ECI-coordinate
        - t: optional time to account for earth
             angular velocity (default = 0)

    Returns:
        - ecef_xyz: [x, y, z] ECEF-coordinates
    ---------------------------------------------------
    """
    eci_xyz = np.asarray(eci_xyz)
    omega_e = 7.2921159e-5  # WGS-84
    dcm_eci_to_ecef = np.array([
        [np.cos(omega_e * t), np.sin(omega_e * t), 0],
        [-np.sin(omega_e * t), np.cos(omega_e * t), 0],  # type:ignore
        [0, 0, 1]])
    return dcm_eci_to_ecef @ eci_xyz


def ecef_to_eci(ecef_xyz: np.ndarray|list, t: float = 0) -> np.ndarray:
    """This method converts a non-velocity vector from ECEF
    coordinates to ECI.

    ---------------------------------------------------
    Arguments:
        - ecef_xyz: [x, y, z] ECEF-coordinate
        - t: optional time to account for earth
             angular velocity (default = 0)

    Returns:
        - eci_xyz: [x, y, z] ECI-coordinates
    ---------------------------------------------------
    """
    ecef_xyz = np.asarray(ecef_xyz)
    omega_e = 7.2921159e-5  # wgs-84
    dcm_eci_to_ecef = np.array([
        [np.cos(omega_e * t), np.sin(omega_e * t), 0],
        [-np.sin(omega_e * t), np.cos(omega_e * t), 0],  # type:ignore
        [0, 0, 1]])
    return dcm_eci_to_ecef.T @ ecef_xyz


# TODO: this may be wrong: see     vel_ecef = C_eci_to_ecef @ _vel - omega_skew_ie @ _pos
def eci_to_ecef_velocity(eci_xyz: np.ndarray|list, r_ecef: np.ndarray|list, t: float) -> np.ndarray:
    """This method converts a velocity vector from ECI
    coordinates to ECEF by taking into account Earth's
    rotation.

    vec_ecef = v_eci − omega_kew * r_ecef

    where omega_skew is a skew matrix and r_ecef is a
    position vector

    ---------------------------------------------------
    Arguments:
        - eci_xyz: [x, y, z] ECI-coordinate
        - r_ecef: [x, y, z] ECEF-coorindate

    Returns:
        - ecef_xyz: [x, y, z] ECEF-coordinates
    ---------------------------------------------------
    """
    eci_xyz = np.asarray(eci_xyz)
    r_ecef = np.asarray(r_ecef)
    omega_e = 7.2921159e-5  # WGS-84
    omega_skew = np.array([
        [0, -omega_e, 0],
        [omega_e, 0, 0],
        [0, 0, 0]])
    C_eci_to_ecef = np.array([
        [np.cos(omega_e * t), np.sin(omega_e * t), 0],
        [-np.sin(omega_e * t), np.cos(omega_e * t), 0],
        [0, 0, 1]])
    return C_eci_to_ecef @ eci_xyz - omega_skew @ r_ecef


def ecef_to_eci_velocity(ecef_xyz: np.ndarray|list, r_ecef: np.ndarray|list) -> np.ndarray:
    """This method converts a velocity vector from ECEF
    coordinates to ECI by taking into account Earth's
    rotation.

    vec_eci = vec_ecef + omega_skew * r_ecef

    where omega_skew is a skew matrix and r_ecef is a
    position vector

    ---------------------------------------------------
    Arguments:
        - ecef_xyz: [x, y, z] ECEF-coordinate
        - r_ecef: [x, y, z] ECEF-coorindate

    Returns:
        - eci_xyz: [x, y, z] ECI-coordinates
    ---------------------------------------------------
    """
    ecef_xyz = np.asarray(ecef_xyz)
    r_ecef = np.asarray(r_ecef)
    omega_e = 7.2921159e-5  # wgs-84
    omega_skew = np.array([
        [0, -omega_e, 0],
        [omega_e, 0, 0],
        [0, 0, 0]])
    return ecef_xyz + omega_skew @ r_ecef


def ecef_to_lla(ecef_xyz: np.ndarray|list) -> np.ndarray:
    """
    convert from ECEF to geodetic
    Olson, D. K. (1996). Converting Earth-Centered,
    Earth-Fixed Coordinates to Geodetic Coordinates.
    IEEE Transactions on Aerospace and Electronic Systems,
    32(1), 473–476. https://doi.org/10.1109/7.481290

    U.S. Government work, U.S. copyright does not apply.

    sa https://ieeexplore.ieee.org/document/481290

    ---------------------------------------------------
    Arguments:
        - ecef_xyz: [ecef-x, ecef-y, ecef-z] coordinates

    Returns:
        - lat: lattitude (radians)
        - lon: longitude (radians)
        - ht: height (meters)
    ---------------------------------------------------
    """
    fabs = np.abs
    sqrt = np.sqrt
    acos = np.arccos
    asin = np.arcsin
    atan2 = np.arctan2

    x, y, z = ecef_xyz

    a = 6378137.0  # wgs-84
    e2 = 6.6943799901377997e-3  # e**2
    a1 = 4.2697672707157535e+4  # a * e2
    a2 = 1.8230912546075455e+9  # a1 * a1
    a3 = 1.4291722289812413e+2  # a1 * e2 / 2
    a4 = 4.5577281365188637e+9  # 2.5 * a2
    a5 = 4.2840589930055659e+4  # a1 + a3
    a6 = 9.9330562000986220e-1  # 1 - e2

    lat = 0
    lon = 0
    ht = 0

    zp = fabs(z)
    w2 = x * x + y * y
    w = sqrt(w2)
    z2 = z * z
    r2 = w2 + z2
    r = sqrt(r2)
    if (r < 100000.0):
        lat = 0.0
        lon = 0.0
        ht = -1.e7
        return np.array([lat, lon, ht])

    lon = atan2(y, x)
    s2 = z2 / r2
    c2 = w2 / r2
    u = a2 / r
    v = a3 - a4 / r

    if (c2 > 0.3):
        s = (zp / r) * (1.0 + c2 * (a1 + u + s2 * v) / r)
        lat = asin(s)
        ss = s * s
        c = sqrt(1.0 - ss)
    else:
        c = (w / r) * (1.0 - s2 * (a5 - u - c2 * v) / r)
        lat = acos(c)
        ss = 1.0 - c * c
        s = sqrt(ss)

    g = 1.0 - e2 * ss
    rg = a / sqrt(g)
    rf = a6 * rg
    u = w - rg * c
    v = zp - rf * s
    f = c * u + s * v
    m = c * v - s * u
    p = m / (rf / g + f)
    lat = lat + p
    ht = f + m * p / 2.0
    if (z < 0.0):
        lat = -lat

    return np.array([lat, lon, ht])


def eci_to_enu_position(eci_xyz: np.ndarray|list, ecef0: Optional[np.ndarray|list],
                        t: float) -> np.ndarray:
    """
    This method converts a position vector from ECI
    coordinates to ENU.

    ---------------------------------------------------
    Arguments:
        - eci_xyz: [x, y, z] ECI-coordinate
        - ecef0: [x0, y0, z0] ECEF-coordinate this is the
                 coordinates for the reference frame origin.
        - t: elapsed time (seconds)

    Returns:
        - enu: [east, north, up] ENU-coordinates
    ---------------------------------------------------
    """
    ecef0 = np.asarray(ecef0)
    ecef = eci_to_ecef(eci_xyz, t=t)
    enu = ecef_to_enu_position(ecef, ecef0)
    return enu


# TODO: i think this is wrong
def eci_to_enu_velocity(eci_xyz: np.ndarray|list, r_ecef: np.ndarray|list,
                        t: float) -> np.ndarray:
    """
    This method converts a velocity vector from ECI coordinates
    to ENU.

    ---------------------------------------------------
    Arguments:
        - eci_xyz: [x, y, z] ECI-coordinate
        - r_ecef: [x, y, z] ECEF-coordinate position vector
        - ecef0: [x0, y0, z0] ECEF-coordinate reference vector
                  this is the coordinates for the reference
                  frame origin.

    Returns:
        - enu: [east, north, up] ENU-coordinates
    ---------------------------------------------------
    """
    r_ecef = np.asarray(r_ecef)
    ecef = eci_to_ecef_velocity(eci_xyz, r_ecef=r_ecef, t=t)
    enu = ecef_to_enu(ecef, r_ecef)
    return enu


def eci_to_enu(eci_xyz: np.ndarray|list, ecef0: np.ndarray|list, t: float = 0) -> np.ndarray:
    """
    This method converts a vector from ECI coordinates
    to ENU.

    ---------------------------------------------------
    Arguments:
        - eci_xyz: [x, y, z] ECI-coordinate
        - t: elapsed time (seconds)

    Returns:
        - enu: [east, north, up] ENU-coordinates
    ---------------------------------------------------
    """
    ecef = eci_to_ecef(eci_xyz, t=t)
    enu = ecef_to_enu(ecef, ecef0)
    return enu


# Convert reference location from LLA to ECEF
def lla_to_ecef(lla):
    """
    This method converts from LLA coordinates to ECEF.

    ---------------------------------------------------
    Arguments:
        - lla: [lat, lon, alt] LLA-coordinate

    Returns:
        - ecef_xyz: [x, y, z] ECEF-coordinates
    ---------------------------------------------------
    """
    lat, lon, h = lla
    e2 = Earth.eccentricity**2
    a = Earth.semimajor_axis
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = ((1 - e2) * N + h) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_enu_position(ecef_xyz: np.ndarray|list, ecef0: np.ndarray|list) -> np.ndarray:
    """
    This method converts a position vector from ECEF
    coordinates to ENU.

    ---------------------------------------------------
    Arguments:
        - ecef_xyz: [x, y, z] ECEF-coordinate
        - ecef0: [x0, y0, z0] ECEF-reference point. This
                 determines the origin of the local ENU
                 frame.

    Returns:
        - enu: [east, north, up] ENU-coordinates
    ---------------------------------------------------
    """
    ecef0 = np.asarray(ecef0)
    cos = np.cos
    sin = np.sin
    lat, lon, alt = ecef_to_lla(ecef0)
    dcm_ecef_to_enu = np.array([[-sin(lon), cos(lon), 0],
                                [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
                                [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]])
    return dcm_ecef_to_enu @ (ecef_xyz - ecef0)


# TODO: this needed fixing
def ecef_to_enu(ecef_xyz: np.ndarray|list, r_ecef: np.ndarray|list) -> np.ndarray:
    """
    This method converts a vector from ECEF coordinates
    to ENU.

    ---------------------------------------------------
    Arguments:
        - ecef_xyz: [x, y, z] ECEF-coordinate
        - r_ecef: [x0, y0, z0] ECEF-reference point. This
                 determines the origin of the local ENU
                 frame.

    Returns:
        - enu: [east, north, up] ENU-coordinates
    ---------------------------------------------------
    """
    r_ecef = np.asarray(r_ecef)
    cos = np.cos
    sin = np.sin
    lat, lon, alt = ecef_to_lla(r_ecef)
    dcm_ecef_to_enu = np.array([[-sin(lon), cos(lon), 0],
                                [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
                                [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]])
    return dcm_ecef_to_enu @ ecef_xyz


def enu_to_ecef_position(enu: np.ndarray|list, ecef0: np.ndarray|list):
    """
    This method converts a position vector from ENU
    coordinates to ECEF.

    ---------------------------------------------------
    Arguments:
        - enu: [east, north, up] ENU-coordinate
        - ecef0: [x0, y0, z0] ECEF-reference point. This
                 determines the origin of the local ENU
                 frame.

    Returns:
        - ecef_xyz: [x, y, z] ECEF-coordinates
    ---------------------------------------------------
    """
    ecef0 = np.asarray(ecef0)
    cos = np.cos
    sin = np.sin
    lat, lon, alt = ecef_to_lla(ecef0)
    dcm_ecef_to_enu = np.array([[-sin(lon), cos(lon), 0],
                                [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
                                [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]])
    return (dcm_ecef_to_enu.T @ enu) + ecef0


def enu_to_ecef(enu: np.ndarray|list, ecef0: np.ndarray|list):
    """
    This method converts a vector from ENU coordinates
    to ECEF.

    ---------------------------------------------------
    Arguments:
        - enu: [east, north, up] ENU-coordinate
        - ecef0: [x0, y0, z0] ECEF-reference point. This
                 determines the origin of the local ENU
                 frame.

    Returns:
        - ecef_xyz: [x, y, z] ECEF-coordinates
    ---------------------------------------------------
    """
    ecef0 = np.asarray(ecef0)
    cos = np.cos
    sin = np.sin
    lat, lon, alt = ecef_to_lla(ecef0)
    dcm_ecef_to_enu = np.array([[-sin(lon), cos(lon), 0],
                                [-sin(lat) * cos(lon), -sin(lat) * sin(lon), cos(lat)],
                                [cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)]])
    return dcm_ecef_to_enu.T @ enu


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
