from sympy import symbols, Matrix, MatrixSymbol, Piecewise, Expr
from sympy import Max, Min, Abs
from sympy import sin, cos, asin, atan2



# This files contains methods which mirror Rotation.py
# but are symbolically defined; returning sympy expressions.

def quat_conj_sym(q: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    assert q.shape == (4, 1)
    return Matrix([q[0], -q[1], -q[2], -q[3]]) #type:ignore


def quat_mult_sym(q: Matrix|MatrixSymbol, p: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    if isinstance(p, MatrixSymbol):
        p = p.as_mutable()
    return Matrix([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3], #type:ignore
                   p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2], #type:ignore
                   p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1], #type:ignore
                   p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]]) #type:ignore


def quat_to_dcm_sym(q: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    assert q.shape == (4, 1)
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = MatrixSymbol("dcm", 3, 3).as_mutable()
    dcm[0, 0] = q0**2 + q1**2 - q2**2 - q3**2   #type:ignore
    dcm[0, 1] = 2.0 * (q1*q2 - q0*q3)           #type:ignore
    dcm[0, 2] = 2.0 * (q1*q3 + q0*q2)           #type:ignore
    dcm[1, 0] = 2.0 * (q1*q2 + q0*q3)           #type:ignore
    dcm[1, 1] = q0**2 - q1**2 + q2**2 - q3**2   #type:ignore
    dcm[1, 2] = 2.0 * (q2*q3 - q0*q1)           #type:ignore
    dcm[2, 0] = 2.0 * (q1*q3 - q0*q2)           #type:ignore
    dcm[2, 1] = 2.0 * (q2*q3 + q0*q1)           #type:ignore
    dcm[2, 2] = q0**2 - q1**2 - q2**2 + q3**2   #type:ignore

    # This form removes q1 from the 0,0, q2 from the 1,1 and q3 from the 2,2 entry and results
    # in a covariance prediction that is better conditioned.
    # It requires the quaternion to be unit length and is mathematically identical
    # to the alternate form when q0**2 + q1**2 + q2**2 + q3**2 = 1
    # See https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    # dcm = Matrix([[1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3)    , 2*(q1*q3 + q0*q2)    ],
    #              [2*(q1*q2 + q0*q3)     , 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)    ],
    #              [2*(q1*q3-q0*q2)       , 2*(q2*q3 + q0*q1)    , 1 - 2*(q1**2 + q2**2)]])

    return dcm


def quat_to_tait_bryan_sym(q: Matrix|MatrixSymbol) -> Expr:
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
        [cpitch*cyaw, sroll*spitch*cyaw - croll*syaw, sroll*syaw + croll*spitch*cyaw], #type:ignore
        [cpitch*syaw, croll*cyaw + sroll*spitch*syaw, croll*spitch*syaw - sroll*cyaw], #type:ignore
        [-spitch, sroll*cpitch, croll*cpitch], #type:ignore
        ])
    return dcm


def dcm_to_tait_bryan_sym(dcm: Matrix|MatrixSymbol) -> Expr:
    if isinstance(dcm, MatrixSymbol):
        dcm = dcm.as_mutable()
    assert isinstance(dcm, Matrix)
    assert dcm.shape == (3, 3)

    ret1 = Matrix([
        atan2(-dcm[0, 1], -dcm[0, 2]), #type:ignore
        asin(Max(dcm[2, 0], -1.0)), #type:ignore
        0
        ])
    ret2 = Matrix([
        atan2(dcm[0, 1], dcm[0, 2]),
        -asin(Min(dcm[2, 0], 1.0)), #type:ignore
        0
        ])
    ret3 = Matrix([
        atan2(dcm[1, 0], dcm[0, 0]),
        -asin(dcm[2, 0]),            #type:ignore
        atan2(dcm[2, 1], dcm[2, 2])
        ])

    piece1 = Piecewise(
            (ret1, dcm[2, 0] < 0), #type:ignore
            (ret2, True)
            )

    dcm_to_tait_bryan_expr = Piecewise(
            (piece1, Abs(Abs(dcm[2, 0]) - 1) < 1e-15), #type:ignore
            (ret3, True)
            )

    return dcm_to_tait_bryan_expr

