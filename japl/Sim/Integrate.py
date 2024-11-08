import numpy as np
from sympy import Symbol, Matrix
from collections.abc import Callable
from japl.BuildTools.BuildTools import parallel_subs



def runge_kutta_4(f: Callable, t: float, X: np.ndarray, dt: float, args: tuple = ()) -> tuple:
    """
        This method integrates state dynamics using Runge Kutta 4 method.
    and returns the value of the state 'X' for the next time step.

    -------------------------------------------------------------------
    -- Arguments
    -------------------------------------------------------------------
    -- f - function which returns the dynamics of the state. 'f' must
            follow the prototype f(t, X, ...)
    -- t - time for the current time step
    -- X - (N x 1) state array
    -- dt - step size for the integration
    -- args - other arguments required by f()
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    -- Returns:
    -------------------------------------------------------------------
    --- Xnew - state array for the next time step
    --- Tnew - the next time step after integration
    -------------------------------------------------------------------
    """

    k1 = f(t, X, *args).flatten()
    k2 = f(t + 0.5 * dt, X + (0.5 * dt * k1), *args).flatten()
    k3 = f(t + 0.5 * dt, X + (0.5 * dt * k2), *args).flatten()
    k4 = f(t + dt, X + (dt * k3), *args).flatten()
    X_new = X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    T_new = t + dt
    return (X_new, T_new)


def runge_kutta_4_symbolic(f: Matrix, t: Symbol, X: Matrix, dt: Symbol, args: tuple = ()) -> Matrix:
    """
        This method integrates state dynamics using Runge Kutta 4 method.
    and returns the value of the state 'X' for the next time step.

    -------------------------------------------------------------------
    -- Arguments
    -------------------------------------------------------------------
    -- f - function which returns the dynamics of the state. 'f' must
            follow the prototype f(t, X, ...)
    -- t - time for the current time step
    -- X - (N x 1) state array
    -- dt - step size for the integration
    -- args - other arguments required by f()
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    -- Returns:
    -------------------------------------------------------------------
    --- X_new - next state array expresion for the next time step
    -------------------------------------------------------------------
    """

    k1 = f.copy()

    k2_subs = ((t, t + 0.5 * dt),)  # type:ignore
    for x, _k1 in zip(X, k1):  # type:ignore
        k2_subs += ((x, x + (0.5 * dt * _k1)),)  # type:ignore
    k2 = parallel_subs(f, k2_subs)

    k3_subs = ((t, t + 0.5 * dt),)  # type:ignore
    for x, _k2 in zip(X, k2):  # type:ignore
        k3_subs += ((x, x + (0.5 * dt * _k2)),)  # type:ignore
    k3 = parallel_subs(f, k3_subs)

    k4_subs = ((t, t + dt),)  # type:ignore
    for x, _k3 in zip(X, k3):  # type:ignore
        k4_subs += ((x, x + (dt * _k3)),)  # type:ignore
    k4 = parallel_subs(f, k4_subs)

    X_new = X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)  # type:ignore
    return X_new

    # ####################################
    # # TEST
    # ####################################
    # from sympy.abc import x, y, z
    # t = sp.symbols("t")
    # dt = sp.symbols("dt")
    # expr = Matrix([x + 2 * t + 0.5 * 3. * t**2])
    # Xnew, Tnew = runge_kutta_4_symbolic(expr.diff(t), t, expr, dt)

    # ret = Xnew.subs({t: 0., x: 1., y: 2., z: 3., dt: 0.1})[0]
    # print(ret)

    # t = 0
    # dt = 0.1
    # x = 1
    # y = 2
    # z = 3
    # f = lambda t, X, *args: y + z * t**1
    # # X = f(t, 0)
    # Xnew, Tnew = runge_kutta_4(f, t, 1., dt)
    # print(Xnew)
    # ####################################


def euler(f: Callable, t: float, X: np.ndarray, dt: float, args: tuple = ()) -> tuple:
    """
        This method integrates state dynamics using Euler's method.
    and returns the value of the state 'X' for the next time step.

    -------------------------------------------------------------------
    -- Arguments
    -------------------------------------------------------------------
    -- f - function which returns the dynamics of the state. 'f' must
            follow the prototype f(t, X, ...)
    -- t - time for the current time step
    -- X - (N x 1) state array
    -- dt - step size for the integration
    -- args - other arguments required by f()
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    -- Returns:
    -------------------------------------------------------------------
    --- Xnew - state array for the next time step
    --- Tnew - the next time step after integration
    -------------------------------------------------------------------
    """

    X_new = X + f(t, X, *args) * dt
    T_new = t + dt
    return (X_new, T_new)
