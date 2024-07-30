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


def quat_to_dcm_sym(q: Matrix|MatrixSymbol) -> Matrix:
    if isinstance(q, MatrixSymbol):
        q = q.as_mutable()
    assert q.shape == (4, 1)
    q0_2 = q[0] * q[0] #type:ignore
    q1_2 = q[1] * q[1] #type:ignore
    q2_2 = q[2] * q[2] #type:ignore
    q3_2 = q[3] * q[3] #type:ignore
    q1q2 = q[1] * q[2] #type:ignore
    q0q3 = q[0] * q[3] #type:ignore
    q1q3 = q[1] * q[3] #type:ignore
    q0q1 = q[0] * q[1] #type:ignore
    q2q3 = q[2] * q[3] #type:ignore
    q0q2 = q[0] * q[2] #type:ignore

    dcm = MatrixSymbol("dcm", 3, 3).as_mutable()
    dcm[0, 0] = q0_2 + q1_2 - q2_2 - q3_2
    dcm[0, 1] = 2.0 * (q1q2 - q0q3)
    dcm[0, 2] = 2.0 * (q1q3 + q0q2)
    dcm[1, 0] = 2.0 * (q1q2 + q0q3)
    dcm[1, 1] = q0_2 - q1_2 + q2_2 - q3_2
    dcm[1, 2] = 2.0 * (q2q3 - q0q1)
    dcm[2, 0] = 2.0 * (q1q3 - q0q2)
    dcm[2, 1] = 2.0 * (q2q3 + q0q1)
    dcm[2, 2] = q0_2 - q1_2 - q2_2 + q3_2

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
            (piece1, Abs(Abs(dcm[2, 0]) - 1) < 1e-12), #type:ignore
            (ret3, True)
            )

    return dcm_to_tait_bryan_expr

