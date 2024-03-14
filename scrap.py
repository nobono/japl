import numpy as np
import sympy as sp
import control as ct



wapar = 25
zetapar = .1
tf = ct.tf([1], [1/wapar**2, 2*zetapar/wapar, 1])
ss = ct.tf2ss(tf)
ss, TM = ct.observable_form(ss)
# ss.set_states(states=["acc", "jerk"])
# ss.set_inputs(inputs=["acc_cmd"])


A = np.array([
    [0, 1],
    [0, 0],
    ])
B = np.array([
    [0],
    [1],
    ])
C = np.eye(2)
D = np.zeros((2, 1))
basess = ct.ss(A, B, C, D)

ss_app = ct.append(basess, tf)
Q = np.array([
    [1, ],
    [2, ],
    ])
# ret = ct.connect(ss_app, Q, inputv=, outputv=)
ct.interconnect

