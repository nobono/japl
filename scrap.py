from pprint import pprint
import numpy as np
import sympy as sp
from sympy import Matrix, MatrixSymbol
import control as ct
from dynamics import BaseSystem


# Setup base dynamics for vehicle
name        = "missile"
states      = ["xpos", "ypos", "zpos", "xvel", "yvel", "zvel"]
states_dot  = ["xvel", "yvel", "zvel", "xacc", "yacc", "zacc"]
outputs     = ["xpos", "ypos", "zpos", "xvel", "yvel", "zvel"]
inputs      = ["xacc", "yacc", "zacc"]
A = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    ])
B = np.array([
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    ])
C = np.eye(6)
D = np.zeros((6, 3))
SS = BaseSystem(A, B, C, D)
SS.set_states(states)
SS.set_states_dot(states_dot)
SS.set_inputs(inputs)
SS.set_outputs(outputs)



name        = "autopilot"
states      = ["xacc", "xjerk"]
states_dot  = ["xjerk", "xjerk_dot"]
inputs      = ["xacc_cmd"]
outputs     = ["xacc"]

wapar = 25
zetapar = .1
tf = ct.tf([1], [1/wapar**2, 2*zetapar/wapar, 1])
ss = ct.tf2ss(tf)
ss, TM = ct.observable_form(ss)

ss = BaseSystem(ss)
ss.set_states(states)
ss.set_states_dot(states_dot)
ss.set_inputs(inputs)
ss.set_outputs(outputs)



# ss_app = ct.append(SS, ss)
# ss_app.set_states(SS.state_labels + ss.state_labels)
# ss_app.set_inputs(SS.input_labels + ss.input_labels)
# ss_app.set_outputs(SS.output_labels + ss.output_labels)

# Q = np.array([
#     [ss_app.input_index["xacc_cmd"], ],
#     ])
# # ret = ct.connect(ss_app, Q, inputv=, outputv=)
# ct.interconnect([SS, ss],
#                 connections=["dynamics.xacc", "autopilot.xacc_cmd"])
connections = {
        "output": ["xacc"],
        "state": ["xacc"]
        }
SS.add_system(ss, connections=connections)

print(SS.state_dot_labels)
print(SS.state_dot_index)
