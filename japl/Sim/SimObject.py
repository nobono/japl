# ---------------------------------------------------

from typing import Optional

# ---------------------------------------------------

import numpy as np

# ---------------------------------------------------

from control.iosys import StateSpace

# ---------------------------------------------------

import astropy.units as u
from astropy.units.quantity import Quantity

# ---------------------------------------------------

from .UnitCheck import assert_physical_type

# ---------------------------------------------------

from .Model import Model
from .Model import ModelType



class SimObject:

    """This is a base class for simulation objects"""

    def __init__(self, model: Optional[Model] = None, **kwargs) -> None:

        assert isinstance(model, Model)
        self.name = kwargs.get("name", "SimObject")
        self.color = kwargs.get("color")
        self.model = model


    def _pre_sim_checks(self) -> bool:
        assert self.model is not None
        return True


    def step(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        return self.model.step(X, U)
            


# class MassObject(SimObject):

#     """This is a base class for mass objects"""

#     def __init__(self,
#                  mass: Quantity,
#                  x0: Quantity,
#                  v0: Quantity,
#                  ):

#         assert_physical_type(mass, "mass")
#         assert_physical_type(x0, "length")
#         assert_physical_type(v0, "velocity")

#         self.mass = mass
#         self.x0 = x0
#         self.v0 = v0


# class RigidBodyObject(SimObject):

#     """This is a base class for mass objects"""

#     def __init__(self,
#                  mass: Quantity,
#                  inertia: Quantity,
#                  x0: Quantity,
#                  v0: Quantity,
#                  ):

#         assert_physical_type(mass, "mass")
#         assert_physical_type(inertia, "moment of inertia")
#         assert_physical_type(x0, "length")
#         assert_physical_type(v0, "velocity")

#         self.mass = mass
#         self.inertia = inertia
#         self.x0 = x0
#         self.v0 = v0
