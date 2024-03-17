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

            [0, 0, 0,   0, 0, 0,    apss.A[0][0], 0, 0,    apss.A[0][1], 0, 0], # xjerk
            [0, 0, 0,   0, 0, 0,    0, apss.A[0][0], 0,    0, apss.A[0][1], 0], # yjerk
            [0, 0, 0,   0, 0, 0,    0, 0, apss.A[0][0],    0, 0, apss.A[0][1]], # zjerk

            [0, 0, 0,   0, 0, 0,    apss.A[1][0], 0, 0,    apss.A[1][1], 0, 0], # xjerk_dot
            [0, 0, 0,   0, 0, 0,    0, apss.A[1][0], 0,    0, apss.A[1][1], 0], # yjerk_dot
            [0, 0, 0,   0, 0, 0,    0, 0, apss.A[1][0],    0, 0, apss.A[1][1]], # zjerk_dot
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

            [0, 0, 0,   0, 0, 0,    apss.A[0][0], 0, 0], # xjerk
            [0, 0, 0,   0, 0, 0,    0, apss.A[0][0], 0], # yjerk
            [0, 0, 0,   0, 0, 0,    0, 0, apss.A[0][0]], # zjerk
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
        self.state_dot_labels = []
        self.state_dot_index = {}

        if len(args) == 4:
            super().__init__(
                StateSpace(*args, init_namedio=True, name="dynamics")
            )
        elif len(args) == 0:
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


    def set_states_dot(self, states_dot: list[str]):
        self.state_dot_labels = states_dot
        for i, name in enumerate(states_dot):
            self.state_dot_index[name] = i


    def get_init_state(self):
        return np.zeros((self.A.shape[0],)) #type:ignore
    

    def add_system(self, ss: "BaseSystem", connections: dict={}):
        """
        @params:
            ss - system
            connections - mapping of {input: output} connections
        """
        # Check for labels
        assert ss.nstates == len(ss.state_labels)
        assert ss.ninputs == len(ss.input_labels)
        assert ss.noutputs == len(ss.output_labels)

        # Extend to matrices
        new_nstates = self.nstates + ss.nstates
        self.A = np.hstack((self.A, np.zeros((self.nstates, ss.nstates))))
        self.A = np.vstack((self.A, np.zeros((ss.nstates, new_nstates))))
        self.C = np.eye(new_nstates)
        self.B = np.vstack((self.B, np.zeros((ss.nstates, self.ninputs))))
        self.D = np.vstack((self.D, np.zeros((ss.nstates, self.ninputs))))

        # Dimension checks
        assert self.A.shape[0] == self.A.shape[1]
        assert self.A.shape == self.C.shape
        assert self.A.shape[0] == self.B.shape[0]
        assert self.A.shape[0] == self.D.shape[0]
        assert self.B.shape[1] == self.D.shape[1]

        # Append labels
        self.set_states(self.state_labels + ss.state_labels)
        self.set_states_dot(self.state_dot_labels + ss.state_dot_labels)
        self.set_inputs(self.input_labels + ss.input_labels)
        self.set_outputs(self.output_labels + ss.output_labels)

        # Insert to A matrix
        for output in ss.output_index.keys():
            output_id = self.state_dot_index.get(output)
            for state in ss.state_index.keys():
                state_id = self.state_index.get(state)
                self.A[output_id][state_id] = ss.C[]
            if not output_id:
                raise Exception(f"output {output} could not be inserted to state\
                        dynamics matrix. state does not exist.")

            # self.A[dynamics_insert_id] = 

        # make connections
        # for 
