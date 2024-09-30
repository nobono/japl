from sympy import Matrix, MatrixSymbol, Piecewise
from sympy import Max, Min, Abs
from sympy import Float
from sympy import sin, cos, asin, acos, atan2
from sympy import sqrt
from japl.Library.Earth.EarthModelSymbolic import EarthModelSymbolic

# This files contains methods which mirror Rotation.py
# but are symbolically defined; returning sympy expressions.



def Sq(q: Matrix|MatrixSymbol) -> Matrix:
    """quaternion dynamics matrix q_dot = dt * 0.5 * Sq * ang_vel"""
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    assert len(q) == 4
    q0, q1, q2, q3 = q
    return Matrix([[-q1, -q2, -q3],  # type:ignore
                   [q0, -q3, q2],  # type:ignore
                   [q3, q0, -q1],  # type:ignore
                   [-q2, q1, q0]])  # type:ignore


def Sw(ang_vel: Matrix|MatrixSymbol) -> Matrix:
    """quaternion dynamics matrix q_dot = -dt * 0.5 * Sw * q"""
    if isinstance(ang_vel, MatrixSymbol):
        ang_vel = ang_vel.as_mutable()
    assert len(ang_vel) == 3
    wx, wy, wz = ang_vel
    return Matrix([[0, -wx, -wy, -wz],  # type:ignore
                   [wx, 0, wz, -wy],  # type:ignore
                   [wy, -wz, 0, wx],  # type:ignore
                   [wz, wy, -wx, 0]])  # type:ignore



def quat_conj_sym(q: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    assert q.shape == (4, 1)
    return Matrix([q[0], -q[1], -q[2], -q[3]])  # type:ignore


def quat_norm_sym(q: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    return (q / q.norm())


def quat_mult_sym(q: Matrix|MatrixSymbol, p: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    if isinstance(p, MatrixSymbol):
        p = p.as_mutable()
    return Matrix([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],  # type:ignore
                   p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],  # type:ignore
                   p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],  # type:ignore
                   p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]])  # type:ignore


def quat_to_dcm_sym(q: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    assert q.shape == (4, 1)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    # This form removes q1 from the 0,0, q2 from the 1,1 and q3 from the 2,2 entry and results
    # in a covariance prediction that is better conditioned.
    # It requires the quaternion to be unit length and is mathematically identical
    # to the alternate form when q0**2 + q1**2 + q2**2 + q3**2 = 1
    # See https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    dcm = Matrix([[1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3)    , 2*(q1*q3 + q0*q2)    ],   # type:ignore # noqa
                 [2*(q1*q2 + q0*q3)     , 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)    ],   # type:ignore # noqa
                 [2*(q1*q3-q0*q2)       , 2*(q2*q3 + q0*q1)    , 1 - 2*(q1**2 + q2**2)]])  # type:ignore # noqa
    return dcm


def quat_to_tait_bryan_sym(q: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    assert q.shape == (4, 1)
    dcm = quat_to_dcm_sym(q)
    return dcm_to_tait_bryan_sym(dcm)


def tait_bryan_to_dcm_sym(yaw_pitch_roll: Matrix|MatrixSymbol) -> Matrix:
    cyaw = cos(yaw_pitch_roll[0])
    cpitch = cos(yaw_pitch_roll[1])
    croll = cos(yaw_pitch_roll[2])
    syaw = sin(yaw_pitch_roll[0])
    spitch = sin(yaw_pitch_roll[1])
    sroll = sin(yaw_pitch_roll[2])
    dcm = Matrix([
        [cpitch*cyaw, sroll*spitch*cyaw - croll*syaw, sroll*syaw + croll*spitch*cyaw],  # type:ignore # noqa
        [cpitch*syaw, croll*cyaw + sroll*spitch*syaw, croll*spitch*syaw - sroll*cyaw],  # type:ignore # noqa
        [-spitch, sroll*cpitch, croll*cpitch],  # type:ignore # noqa
        ])
    return dcm


def dcm_to_tait_bryan_sym(dcm: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(dcm, MatrixSymbol):
        dcm = dcm.as_mutable()
    assert isinstance(dcm, Matrix)
    assert dcm.shape == (3, 3)

    cond1 = dcm[2, 0] < 0  # type:ignore
    cond2 = Abs(Abs(dcm[2, 0]) - 1) < 1e-15  # type:ignore

    ret1_m11 = atan2(-dcm[0, 1], -dcm[0, 2])  # type:ignore
    ret1_m21 = asin(Max(dcm[2, 0], -1.0))  # type:ignore
    ret1_m31 = 0

    ret2_m11 = atan2(dcm[0, 1], dcm[0, 2])
    ret2_m21 = -asin(Min(dcm[2, 0], 1.0))  # type:ignore
    ret2_m31 = 0

    ret3_m11 = atan2(dcm[1, 0], dcm[0, 0])
    ret3_m21 = -asin(dcm[2, 0])             # type:ignore
    ret3_m31 = atan2(dcm[2, 1], dcm[2, 2])

    piece1_m11 = Piecewise(
            (ret1_m11, cond1),  # type:ignore
            (ret2_m11, True)
            )
    piece1_m21 = Piecewise(
            (ret1_m21, cond1),  # type:ignore
            (ret2_m21, True)
            )
    piece1_m31 = Piecewise(
            (ret1_m31, cond1),  # type:ignore
            (ret2_m31, True)
            )

    dcm_to_tait_bryan_expr_m11 = Piecewise(
            (piece1_m11, cond2),  # type:ignore
            (ret3_m11, True)
            )
    dcm_to_tait_bryan_expr_m21 = Piecewise(
            (piece1_m21, cond2),  # type:ignore
            (ret3_m21, True)
            )
    dcm_to_tait_bryan_expr_m31 = Piecewise(
            (piece1_m31, cond2),  # type:ignore
            (ret3_m31, True)
            )

    dcm_to_tait_bryan_expr = Matrix([dcm_to_tait_bryan_expr_m11,
                                     dcm_to_tait_bryan_expr_m21,
                                     dcm_to_tait_bryan_expr_m31])

    return dcm_to_tait_bryan_expr


def ecef_to_lla_sym(ecef_xyz: Matrix|MatrixSymbol) -> Matrix:
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
    if isinstance(ecef_xyz, MatrixSymbol):
        ecef_xyz = ecef_xyz.as_mutable()

    x, y, z = ecef_xyz

    a = Float(6378137.0)  # wgs-84
    e2 = Float(6.6943799901377997e-3)  # e**2
    a1 = Float(4.2697672707157535e+4)  # a * e2
    a2 = Float(1.8230912546075455e+9)  # a1 * a1
    a3 = Float(1.4291722289812413e+2)  # a1 * e2 / 2
    a4 = Float(4.5577281365188637e+9)  # 2.5 * a2
    a5 = Float(4.2840589930055659e+4)  # a1 + a3
    a6 = Float(9.9330562000986220e-1)  # 1 - e2

    zp = Abs(z)
    w2 = x * x + y * y  # type:ignore
    w = sqrt(w2)
    z2 = z * z  # type:ignore
    r2 = w2 + z2
    r = sqrt(r2)

    cond_r = r < 100000.0  # type:ignore
    ret1_m1 = 0.0
    ret1_m2 = 0.0
    ret1_m3 = -1.e7

    lon = atan2(y, x)
    s2 = z2 / r2
    c2 = w2 / r2
    u = a2 / r
    v = a3 - a4 / r

    cond_c2 = c2 > 0.3  # type:ignore
    piece_s = Piecewise(
            ((zp / r) * (1.0 + c2 * (a1 + u + s2 * v) / r), cond_c2),  # type:ignore
            (sqrt(1.0 - ((w / r) * (1.0 - s2 * (a5 - u - c2 * v) / r))**2), True)  # type:ignore
            )
    piece_lat = Piecewise(
            (asin(piece_s), cond_c2),
            (acos((w / r) * (1.0 - s2 * (a5 - u - c2 * v) / r)), True),  # type:ignore
           )
    piece_ss = Piecewise(
            (((zp / r) * (1.0 + c2 * (a1 + u + s2 * v) / r))**2, cond_c2),  # type:ignore
            (1.0 - ((w / r) * (1.0 - s2 * (a5 - u - c2 * v) / r))**2, True)  # type:ignore
            )
    piece_c = Piecewise(
            (sqrt(1.0 - ((zp / r) * (1.0 + c2 * (a1 + u + s2 * v) / r))**2), cond_c2),  # type:ignore
            ((w / r) * (1.0 - s2 * (a5 - u - c2 * v) / r), True)  # type:ignore
            )

    s = piece_s
    lat = piece_lat
    ss = piece_ss
    c = piece_c

    g = 1.0 - e2 * ss  # type:ignore
    rg = a / sqrt(g)
    rf = a6 * rg
    u = w - rg * c  # type:ignore
    v = zp - rf * s  # type:ignore
    f = c * u + s * v
    m = c * v - s * u
    p = m / (rf / g + f)
    lat = lat + p
    ht = f + m * p / 2.0

    cond_z = z < 0.0  # type:ignore
    piece_lat_out = Piecewise(
            (-lat, cond_z),
            (lat, True)
            )

    ret2_m1 = piece_lat_out
    ret2_m2 = lon
    ret2_m3 = ht

    piece_m1 = Piecewise(
            (ret1_m1, cond_r),
            (ret2_m1, True)
            )
    piece_m2 = Piecewise(
            (ret1_m2, cond_r),
            (ret2_m2, True)
            )
    piece_m3 = Piecewise(
            (ret1_m3, cond_r),
            (ret2_m3, True)
            )
    lla = Matrix([piece_m1, piece_m2, piece_m3])
    return lla
