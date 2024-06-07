import matplotlib.pyplot as plt
import unittest
import control as ct
import numpy as np
from japl import Model
from japl import SimObject



class TestExample(unittest.TestCase):

    def test_example(self):

        Tend = .1
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

        omega = 200
        zeta = .3
        act = ct.tf2ss([1], [1/omega**2, 2*zeta/omega, 1])


        x1 = np.array([0, 0])
        x2 = np.array([0, 0])

        U = [1] * N

        T = []
        Y = []
        Y_vehicle = []
        t = 0

        for i in range(N):

            t += dt
            T += [t]

            # update actuator model
            Y += [act.C @ x1]
            u1 = np.array([U[i]])
            x1_dot = act.A @ x1 + act.B @ u1
            x1 = x1_dot * dt + x1

            # update vehicle model
            x2_dot = model.step(x2, (act.C @ x1))
            x2 = x2_dot * dt + x2
            Y_vehicle += [x2]


        Y = np.asarray(Y)
        Y_vehicle = np.asarray(Y_vehicle)

        print(Y_vehicle[-1, 0])

        # ------------------------

        x1 = np.array([0, 0])
        x2 = np.array([0, 0])

        A1 = act.A
        B1 = act.B
        C1 = act.C

        A2 = np.array([
            [0, 1],
            [0, 0],
            ])
        B2 = np.array([
            [0],
            [1],
            ])

        A = np.block([
            [A1, np.zeros((2, 2))],
            [(B2 @ C1), A2],
            ])

        # plt.plot(T, [1] * N)
        # plt.plot(T, Y[:, 0])
        # plt.figure()
        # plt.plot(T, Y_vehicle[:, 0])
        # plt.figure()
        # plt.plot(T, Y_vehicle[:, 1])
        # plt.show()

                


if __name__ == '__main__':
    unittest.main()
