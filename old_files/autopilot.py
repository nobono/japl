import control as ct
import matplotlib.pyplot as plt
import numpy as np



wapar = 25
zetapar = .1
ss = ct.tf2ss([1], [1/wapar**2, 2*zetapar/wapar, 1])
ss, TM = ct.observable_form(ss)

tau = 0.25
# ss = ct.tf2ss([1], [tau, 1])
# ss, TM = ct.observable_form(ss)


if __name__ == "__main__":
    # ss_model = ss
    # t = np.linspace(0, 10, 1000)
    # x0 = np.array([0])
    # U = [np.sin(i) for i in t]
    # # U = [1] * 1000
    # sol = ct.forced_response(ss_model, t, U, x0, return_x=True) #type:ignore
    # tt = sol[0]
    # y = sol[1]
    # s = sol[2].T
    # plt.figure()
    # plt.plot(t, y)
    # plt.plot(t, U, '--')
    # plt.show()

    # TEMP
    #############
    A = np.array([
        [0, 1],
        [0, 0],
        ])
    B = np.array([
        [0],
        [1],
        ])
    C = np.eye(2)
    D = np.array([
        [0],
        [0],
        ])
    ss = ct.ss(A, B, C, D)

    x = np.array([0, 1])
    # u = np.array([1])

    dt = 0.01

    T = np.linspace(0, 10, 100)
    Y = []

    def f(ss):
        return (ss.A @ x + ss.B @ u) * dt + x

    for t in T:
        u = np.array([np.sin(.5 * t)])
        Y += [x]
        x = f(ss)

    Y = np.asarray(Y)

    # plt.plot(T, Y[:, 0])
    plt.plot(T, np.sin(.5 * T))
    plt.show()

