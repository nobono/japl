import numpy as np



def runge_kutta_4(Xdot: np.ndarray, X: np.ndarray, t: float, h: float) -> np.ndarray:
    """
        This method integrates state dynamics using Runge Kutta 4 method.
    and returns the value of the state 'X' for the next time step.
    
    * This method currently does not support non-autonomous systems where
    time is a part of the state dynamics.*

    -------------------------------------------------------------------
    -- Arguments
    -------------------------------------------------------------------
    -- Xdot - (N x 1) state dynamics array
    -- X - (N x 1) state array
    -- t - time for the current time step
    -- h - step size for the integration (dt)
    -------------------------------------------------------------------
    -------------------------------------------------------------------
    -- Returns:
    -------------------------------------------------------------------
    --- Xnew - state array for the next time step
    -------------------------------------------------------------------
    """

    k1 = Xdot
    k2 = X + (0.5 * h * k1)
    k3 = X + (0.5 * h * k2)
    k4 = X + (h * k3)
    return X + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
