# ---------------------------------------------------

from typing import Optional

# ---------------------------------------------------

import numpy as np

# ---------------------------------------------------

import astropy.units as u
from astropy.units.quantity import Quantity

# ---------------------------------------------------

from japl.Util.UnitCheck import assert_physical_type
from japl.Model.Model import Model
from japl.Model.Model import ModelType
from japl.Util.Util import flatten_list

# ---------------------------------------------------

from matplotlib.patches import Circle
from matplotlib.lines import Line2D



class PlotInterface:

    """This is a class for interfacing SimObject data with the plotter."""

    def __init__(self, size: float, state_select: dict, color: Optional[str] = None) -> None:
        self.patch = Circle((0, 0), radius=size, color=color)
        self.trace = Line2D([0], [0], color=color)
        self.state_select = state_select


class SimObject:

    """This is a base class for simulation objects"""

    def __init__(self, model: Model = Model(), **kwargs) -> None:

        assert isinstance(model, Model)
        self.name = kwargs.get("name", "SimObject")
        self.color = kwargs.get("color")
        self.size = kwargs.get("size", 1)
        self.model = model
        self.register = {}
        self.state_dim = self.model.state_dim
        self.X0 = np.zeros((self.state_dim,))
        self.Y = np.array([])
        self.__T = np.array([])

        self.plot = PlotInterface(
                state_select={},
                size=self.size,
                color=self.color
                )


    def _pre_sim_checks(self) -> bool:

        msg_header = f"SimObject \"{self.name}\""

        # check if model exists
        if self.model is None:
            raise AssertionError(f"{msg_header} has no model")

        # check state / output register
        if len(self.register) != self.model.A.shape[0]:
            raise AssertionError(f"{msg_header} register ill-configured")

        # check initial state aray
        if len(self.X0) != self.model.A.shape[0]:
            raise AssertionError(f"{msg_header} initial state \"X0\" ill-configured")

        return True


    def step(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        # TODO: accounting for model inputs here?
        self.update(X)
        return self.model.step(X, U)


    def update(self, X: np.ndarray):
        pass


    def register_state(self, name: str, id: int, label: str = "") -> None:
        self.register.update({name: {"id": id, "label": label}})


    def init_state(self, state: np.ndarray|list) -> None:
        if isinstance(state, list):
            state = flatten_list(state)
        self.X0 = np.asarray(state).flatten()


    def _output_data(self, y) -> None:
        """stores solution data from solver into sim object"""
        self.Y = y


    def get_plot_data(self, index: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """This method returns state data from the SimObject according
        to the user specified state_select."""

        if not self.plot.state_select:
            Warning(f"No state_select configuration set for SimObject \"{self.name}\".")
            return (np.array([]), np.array([]))

        return (self._get_data(index, self.plot.state_select["xaxis"]),
                self._get_data(index, self.plot.state_select["yaxis"]))


    def _get_data(self, index: Optional[int], state_slice: tuple[int, int]|int|str) -> np.ndarray:
        """This method returns state data from the SimObject."""

        if index is None:
            index = len(self.Y) - 1

        if isinstance(state_slice, tuple) or isinstance(state_slice, list):
            return self.Y[:index, state_slice[0]:state_slice[1]]
        elif isinstance(state_slice, int):
            return self.Y[:index, state_slice]
        elif isinstance(state_slice, str):
            if state_slice.lower() in ['t', 'time']:
                return self.__T[:index]
            elif state_slice in self.register:
                return self.Y[:index, self.register[state_slice]["id"]]
            else:
                return np.array([])
        else:
            return np.array([])
            

    def _set_T_array_ref(self, _T) -> None:
        """This method is used to reference the internal __T time array to the 
        Sim class Time array 'T'. This method exists to avoid redundant time arrays in
        various SimObjects."""

        self.__T = _T


    def _update_patch_data(self, xdata: np.ndarray, ydata: np.ndarray) -> None:

        # update trace data
        self.plot.trace.set_data(xdata, ydata)

        # plot current step position data
        self.plot.patch.set_center((xdata[-1], ydata[-1]))


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
