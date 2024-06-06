import unittest
import control as ct
import numpy as np
from japl import Model
from japl import SimObject



class TestExample(unittest.TestCase):

    def test_example(self):

        Tend = 1
        dt = 0.001
        N = int(Tend / dt)

        A = np.array([
            [0, 1],
            [0, 0],
            ])
        B = np.array([
            [0],
            [1],
            ])

        model = Model.ss(A, B)

        omega = 20
        zeta = .8
        act = ct.tf2ss([1], [1/omega**2, 2*zeta/omega, 1])


        x1 = np.array([0, 0])
        x2 = np.array([0, 0])

        U = [3] * (N//2) + [-6] * (N//2)
        # _t = np.linspace(0, Tend, N)
        # U = np.sin(150*_t)

        T = []
        Y = []
        Y_vehicle = []
        t = 0

        for i in range(N):

            t += dt
            T += [t]

            # update actuator model
            Y += [act.C @ x1]
            x_dot = (act.A @ x1 + act.B @ np.array([U[i]]))
            x = x_dot * dt + x1

            # update vehicle model
            X_dot = model.step(x2, (act.C @ x))
            x2 = X_dot * dt + x2
            Y_vehicle += [x2]


        Y = np.asarray(Y)
        Y_vehicle = np.asarray(Y_vehicle)

                


if __name__ == '__main__':
    unittest.main()
