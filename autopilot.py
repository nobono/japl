import control as ct
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.base import OdeSolver
import matplotlib.pyplot as plt
import scipy
import numpy as np



wapar = 25
zetapar = .2
ss = ct.tf2ss([1], [1/wapar**2, 2*zetapar/wapar, 1])
ss, TM = ct.observable_form(ss)


if __name__ == "__main__":
    t = np.linspace(0, 10, 1000)
    x0 = np.array([0, 0])
    U = [np.sin(i) for i in t]
    sol = ct.forced_response(ss, t, U, x0, return_x=True) #type:ignore
    tt = sol[0]
    y = sol[1]
    s = sol[2].T
    plt.figure()
    plt.plot(t, y)
    plt.plot(t, U, '--')
    plt.show()

