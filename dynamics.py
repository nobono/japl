import numpy as np
from autopilot import ss as apss
import control as ct
from control.statesp import StateSpace
from control.iosys import LinearIOSystem



# Second-Order control input
class SecondOrderInput(LinearIOSystem):

    def __init__(self, *args, **kwargs):
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
        super().__init__(StateSpace(A, B, C, D, *args, init_namedio=True, **kwargs))


    def get_init_state(self):
        return np.zeros((self.A.shape[0],)) #type:ignore


# First-Order control input
class FirstOrderInput(LinearIOSystem):

    def __init__(self, *args, **kwargs):
        A = np.array([
            [0, 0, 0,   1, 0, 0,    0, 0, 0], # xvel
            [0, 0, 0,   0, 1, 0,    0, 0, 0], # yvel
            [0, 0, 0,   0, 0, 1,    0, 0, 0], # zvel

            [0, 0, 0,   0, 0, 0,    apss.C[0][0], 0, 0], # xacc
            [0, 0, 0,   0, 0, 0,    0, apss.C[0][0], 0], # yacc
            [0, 0, 0,   0, 0, 0,    0, 0, apss.C[0][0]], # zacc

            [0, 0, 0,   0, 0, 0,    apss.A[0][0], 0, 0], # xacc_cmd
            [0, 0, 0,   0, 0, 0,    0, apss.A[0][0], 0], # yacc_cmd
            [0, 0, 0,   0, 0, 0,    0, 0, apss.A[0][0]], # zacc_cmd
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
            ])
        ct.ss

        C = np.eye(9)
        D = np.zeros((9, 6))
        super().__init__(StateSpace(A, B, C, D, *args, init_namedio=True, **kwargs))


    def get_init_state(self):
        return np.zeros((self.A.shape[0],)) #type:ignore



class BaseSystem(LinearIOSystem):

    def __init__(self, *args, **kwargs):
        A = np.array([
            [0, 0, 0,   1, 0, 0], # xvel
            [0, 0, 0,   0, 1, 0], # yvel
            [0, 0, 0,   0, 0, 1], # zvel

            [0, 0, 0,   0, 0, 0], # xacc
            [0, 0, 0,   0, 0, 0], # yacc
            [0, 0, 0,   0, 0, 0], # zacc
            ])

        # [ax, ay, az]
        # inputs add columns
        B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            ])
        ct.ss

        C = np.eye(6)
        D = np.zeros((6, 3))
        state_labels = ["xpos", "ypos", "zpos", "xvel", "yvel", "zvel"]
        output_labels = ["xpos", "ypos", "zpos", "xvel", "yvel", "zvel"]
        input_labels = ["xacc", "yacc", "zacc"]
        super().__init__(StateSpace(A, B, C, D, *args, init_namedio=True,
                                    states=state_labels,
                                    inputs=input_labels,
                                    outputs=output_labels,
                                    name="dynamics",
                                    ))


    def get_init_state(self):
        return np.zeros((self.A.shape[0],)) #type:ignore
    

    def add_system(self, ss):
        state_labels = self.state_labels
        input_labels = self.input_labels
        output_labels = self.output_labels
        self = ct.append(self, ss)
        self.set_states(state_labels + ss.state_labels)
        self.set_inputs(input_labels + ss.input_labels)
        self.set_outputs(output_labels + ss.output_labels)
