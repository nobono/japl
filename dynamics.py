import numpy as np
from autopilot import ss as apss
import control as ct



A = np.array([
    [0, 0, 0,   1, 0, 0,    0, 0, 0,    0, 0, 0], # xvel
    [0, 0, 0,   0, 1, 0,    0, 0, 0,    0, 0, 0], # yvel
    [0, 0, 0,   0, 0, 1,    0, 0, 0,    0, 0, 0], # zvel

    [0, 0, 0,   0, 0, 0,    apss.C[0][0], 0, 0,    apss.C[0][1], 0, 0], # xacc
    [0, 0, 0,   0, 0, 0,    0, apss.C[0][0], 0,    0, apss.C[0][1], 0], # yacc
    [0, 0, 0,   0, 0, 0,    0, 0, apss.C[0][0],    0, 0, apss.C[0][1]], # zacc

    [0, 0, 0,   0, 0, 0,    apss.A[0][0], 0, 0,    apss.A[0][1], 0, 0], # xacc_cmd
    [0, 0, 0,   0, 0, 0,    0, apss.A[0][0], 0,    0, apss.A[0][1], 0], # yacc_cmd
    [0, 0, 0,   0, 0, 0,    0, 0, apss.A[0][0],    0, 0, apss.A[0][1]], # zacc_cmd

    [0, 0, 0,   0, 0, 0,    apss.A[1][0], 0, 0,    apss.A[1][1], 0, 0], # xacc_cmd_dot
    [0, 0, 0,   0, 0, 0,    0, apss.A[1][0], 0,    0, apss.A[1][1], 0], # yacc_cmd_dot
    [0, 0, 0,   0, 0, 0,    0, 0, apss.A[1][0],    0, 0, apss.A[1][1]], # zacc_cmd_dot
    ])

# [ax, ay, az, ux, uy, uz]
B = np.array([
    [0, 0, 0,   0, 0, 0],
    [0, 0, 0,   0, 0, 0],
    [0, 0, 0,   0, 0, 0],
    [1, 0, 0,   0, 0, 0],
    [0, 1, 0,   0, 0, 0],
    [0, 0, 1,   0, 0, 0],
    [0, 0, 0,   *apss.B[0], 0, 0],
    [0, 0, 0,   0, *apss.B[0], 0],
    [0, 0, 0,   0, 0, *apss.B[0]],
    [0, 0, 0,   *apss.B[1], 0, 0],
    [0, 0, 0,   0, *apss.B[1], 0],
    [0, 0, 0,   0, 0, *apss.B[1]],
    ])

C = np.eye(12)

D = np.zeros((12, 6))

ss = ct.ss(A, B, C, D)
