import control as ct
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.base import OdeSolver
import matplotlib.pyplot as plt
import scipy
import numpy as np



wapar = 3
zetapar = .6
ss = ct.tf2ss([1], [1/wapar**2, 2*zetapar/wapar, 1])
ss, TM = ct.observable_form(ss)


if __name__ == "__main__":
    t = np.linspace(0, 5, 1000)
    sol = ct.step_response(ss, t, return_x=True)
    tt = sol[0]
    y = sol[1]
    s = sol[2].T
    plt.figure()
    plt.plot(t, y)
    plt.show()

