import numpy as np
import sympy as sp
from sympy import Matrix, MatrixSymbol
import control as ct
from dynamics import BaseSystem



wapar = 25
zetapar = .1
tf = ct.tf([1], [1/wapar**2, 2*zetapar/wapar, 1],
           inputs=["xacc_cmd"], outputs=["xacc"], name="autopilot")
ss = ct.tf2ss(tf, states=["xacc", "xjerk"], name="autopilot")
ss, TM = ct.observable_form(ss)
# ss.set_states(states=["acc", "jerk"])
# ss.set_inputs(inputs=["acc_cmd"])


# pos = MatrixSymbol("acc", 3, 1)
# vel = MatrixSymbol("acc", 3, 1)
# ext_acc = MatrixSymbol("acc", 3, 1)
SS = BaseSystem()

# A = np.array([
#     [0, 1],
#     [0, 0],
#     ])
# B = np.array([
#     [0],
#     [1],
#     ])
# C = np.eye(2)
# D = np.zeros((2, 1))
# basess = ct.ss(A, B, C, D,
#                states=["pos", "vel"])

ss_app = ct.append(SS, ss)
ss_app.set_states(SS.state_labels + ss.state_labels)
ss_app.set_inputs(SS.input_labels + ss.input_labels)
ss_app.set_outputs(SS.output_labels + ss.output_labels)

Q = np.array([
    [ss_app.input_index["xacc_cmd"], ],
    ])
# ret = ct.connect(ss_app, Q, inputv=, outputv=)
ct.interconnect([SS, ss],
                connections=["dynamics.xacc", "autopilot.xacc_cmd"])

