import control as ct
import matplotlib.pyplot as plt
import numpy as np
from japl import Model
from japl import SimObject



Tend = 1
dt = 0.001
N = int(Tend / dt)

omega = 20
zeta = .8
ss = ct.tf2ss([1], [1/omega**2, 2*zeta/omega, 1])
# ss_obs, TM = ct.observable_form(ss)

# tau = 0.1
# ss = ct.tf2ss([1], [tau, 1])
# ss_obs, TM = ct.observable_form(ss)

A = np.array([
    [0, 1],
    [0, 0],
    ])
B = np.array([
    [0],
    [1],
    ])

vehicle = Model.ss(A, B)


if __name__ == "__main__":

    x = np.array([0, 0])
    X = np.array([0, 0])

    U = [3] * (N//2) + [-6] * (N//2)
    # _t = np.linspace(0, Tend, N)
    # U = np.sin(150*_t)

    T = []
    Y = []
    Y_vehicle = []
    t = 0


    def f(ss, x, u):
        return (ss.A @ x + ss.B @ u)


    for i in range(N):

        t += dt
        T += [t]

        # update actuator model
        Y += [ss.C @ x]
        x_dot = f(ss, x, np.array([U[i]]))
        x = x_dot * dt + x

        # update vehicle model
        X_dot = vehicle.step(X, (ss.C @ x))
        X = X_dot * dt + X
        Y_vehicle += [X]


    Y = np.asarray(Y)
    Y_vehicle = np.asarray(Y_vehicle)


    plt.plot(T, U)
    plt.plot(T, Y[:, 0], linestyle='-.')

    plt.figure()
    plt.plot(T, Y_vehicle[:, 0])

    plt.show()

