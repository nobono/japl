# ---------------------------------------------------

from typing import Optional

# ---------------------------------------------------

import numpy as np

# ---------------------------------------------------

# from control.iosys import StateSpace

# ---------------------------------------------------

import astropy.units as u
from astropy.units.quantity import Quantity

# ---------------------------------------------------

from japl.Util.UnitCheck import assert_physical_type

# ---------------------------------------------------

from japl.Model.Model import Model
from japl.Model.Model import ModelType

from japl.Util.Util import flatten_list



class SimObject:

    """This is a base class for simulation objects"""

    def __init__(self, model: Model = Model(), **kwargs) -> None:

        assert isinstance(model, Model)
        self.name = kwargs.get("name", "SimObject")
        self.color = kwargs.get("color")
        self.model = model
        self.register = {}
        self.state_dim = self.model.state_dim
        self.X0 = np.zeros((self.state_dim,))
        self.T = np.array([])
        self.Y = np.array([])


    def _pre_sim_checks(self) -> bool:

        msg_header = f"SimObject \"{self.name}\""

        if self.model is None:
            raise AssertionError(f"{msg_header} has no model")

        if len(self.register) != self.model.A.shape[0]:
            raise AssertionError(f"{msg_header} register ill-configured")

        if len(self.X0) != self.model.A.shape[0]:
            raise AssertionError(f"{msg_header} initial state \"X0\" ill-configured")

        return True


    def step(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        # TODO: accounting for model inputs here?
        return self.model.step(X, U)


    def register_state(self, name: str, id: int, label: str = "") -> None:
        self.register.update({name: {"id": id, "label": label}})


    def init_state(self, state: np.ndarray|list) -> None:
        if isinstance(state, list):
            state = flatten_list(state)
        self.X0 = np.asarray(state).flatten()


    def _output_data(self, t, y) -> None:
        """stores solution data from solver into sim object"""
        self.T = t
        self.Y = y
            


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
