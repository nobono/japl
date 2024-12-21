import numpy as np
import lin
from japl import Sim

s1 = lin.SimObject()
s2 = lin.SimObject()

s1.init_state([0, 0])
s2.init_state([0, 0])

s1.children_pre_update += [s2]


def input_func(t, X, U, S, dt, *args):
    return np.array([1.])


s1.model.set_input_function(input_func)
sim = Sim(t_span=[0, 1], dt=0.1, simobjs=[s1])
sim.run()

print(s1.Y)

# t = 0
# dt = 0.1
# X = [0.] * 2
# U = [1]
# S = []
# print(s1.model.dynamics(t, X, U, S, dt))
