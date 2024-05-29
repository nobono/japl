# ---------------------------------------------------

from typing import Optional

# ---------------------------------------------------

import numpy as np

# ---------------------------------------------------

from control.iosys import StateSpace

# ---------------------------------------------------

from astropy import units as u
from astropy.units.quantity import Quantity

# ---------------------------------------------------



class SimObject:

    """This is a base class for simulation objects"""

    def __init__(self,
                 mass: Quantity,
                 inertia: Quantity,
                 x0: Quantity,
                 v0: Quantity = np.zeros((3,)) * (u.m / u.s)):
        pass


s = SimObject(1*u.kg, 1 * u.s, 1*u.s)
