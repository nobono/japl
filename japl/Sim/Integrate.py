import numpy as np
from collections.abc import Callable



def runge_kutta_4(f: Callable, t: float, X: np.ndarray, h: float, args: tuple = ()) -> tuple:
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
    -- h - step size for the integration (dt)
    -- args - other arguments required by f()
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    -- Returns:
    -------------------------------------------------------------------
    --- Xnew - state array for the next time step
    --- Tnew - the next time step after integration
    -------------------------------------------------------------------
    """

    k1 = f(t, X, *args)
    k2 = f(t + 0.5 * h, X + (0.5 * h * k1), *args)
    k3 = f(t + 0.5 * h, X + (0.5 * h * k2), *args)
    k4 = f(t + h, X + (h * k3), *args)
    X_new = X + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    T_new = t + h
    return (X_new, T_new)


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
