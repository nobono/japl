# ---------------------------------------------------

from collections.abc import Generator
from typing import Optional

# ---------------------------------------------------

from matplotlib.axes import Axes
import numpy as np

# ---------------------------------------------------

import astropy.units as u
from astropy.units.quantity import Quantity

from pyqtgraph import GraphicsView, PlotCurveItem, mkPen
from pyqtgraph import CircleROI
from pyqtgraph.Qt.QtGui import QPen

# ---------------------------------------------------

import japl
from japl.Aero.AeroTable import AeroTable
from japl.Util.UnitCheck import assert_physical_type
from japl.Model.Model import Model
from japl.Model.Model import ModelType
from japl.Util.Util import flatten_list

# ---------------------------------------------------

from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from matplotlib import colors as mplcolors



# class ShapeCollection:
#     
#     """This is a class which abstracts the line / shape plots of different
#     plotting backends."""

#     def __init__(self, color: str, radius: float) -> None:
#         # assert plotting_backend in ["matplotlib", "pyqtgraph", "mpl", "qt"]
#         # self.plotting_backend = plotting_backend
#         self.color = color
#         self.radius = radius

#     
#     def setup(self):
#         self.patch = Circle((0, 0), radius=size, color=color)
#         self.trace = Line2D([0], [0], color=color)


class PlotInterface:

    """This is a class for interfacing SimObject data with the plotter."""

    def __init__(self, size: float, state_select: dict, color: Optional[str] = None) -> None:
        self.size = size

        if not color:
            self.color = next(self.color_cycle)
        else:
            self.color = color
        self.color_code = self.get_mpl_color_code(self.color)

        # color cycle list
        self.color_cycle = self.__color_cycle()
        self.__state_select = state_select
        self.plotting_backend = japl.get_plotlib()

        # graphic objects
        self.traces: list[Line2D] = []
        self.qt_traces: list[PlotCurveItem] = []


    def set_config(self, plot_config: dict) -> None:
        # TODO check format here
        self.__state_select = plot_config


    def get_config(self) -> dict:
        return self.__state_select


    def get_mpl_color_code(self, color_str: str = "") -> str:
        color_code = mplcolors.TABLEAU_COLORS[color_str]
        return str(color_code)


    def __color_cycle(self) -> Generator[str, None, None]:
        """This method is a Generator which handles the color cycle of line / scatter
        plots which do not specify a color."""

        while True:
            for _, v in mplcolors.TABLEAU_COLORS.items():
                yield str(v)


    def _get_qt_pen(self, subplot_id: int) -> QPen:
        pen_color = self.qt_traces[subplot_id].opts['pen'].color().getRgb()[:3]
        pen_width = self.qt_traces[subplot_id].opts['pen'].width()
        return mkPen(pen_color, width=pen_width)


    def _update_patch_data(self, xdata: np.ndarray, ydata: np.ndarray, subplot_id: int, **kwargs) -> None:
        if self.plotting_backend == "matplotlib":

            # update trace data
            self.traces[subplot_id].set_data(xdata, ydata)

            # plot current step position data
            # self.patch.set_center((xdata[-1], ydata[-1]))

        if self.plotting_backend == "pyqtgraph":
            self.qt_traces[subplot_id].setData(x=xdata, y=ydata, **kwargs)


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

        self.aerotable: Optional[AeroTable] = kwargs.get("aerotable", None)

        # other physical properties
        self.mass: float = 1
        self.Ixx: float = 1
        self.Iyy: float = 1
        self.Izz: float = 1
        self.cg: float = 1

        # interface for visualization
        self.plot = PlotInterface(
                state_select={},
                size=self.size,
                color=self.color
                )


    def get_state_id(self, name: str) -> int:
        return self.register[name]["id"]


    def _pre_sim_checks(self) -> bool:
        """This method is used to check user configuration of SimObjects
        before running a simulation."""

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
        """This method is the update-step of the SimObject dynamic model. It calls
        the SimObject Model's step() function.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- X - current state array of SimObject
        -- U - current input array of SimObject
        -------------------------------------------------------------------
        -------------------------------------------------------------------
        -- Returns:
        -------------------------------------------------------------------
        -- X_dot - state dynamics "Xdot = A*X + B*U"
        -------------------------------------------------------------------
        """
        # TODO: accounting for model inputs here?
        self.update(X)
        return self.model.step(X, U)


    def update(self, X: np.ndarray):
        pass


    def register_state(self, name: str, id: int, label: str = "") -> None:
        """This method registers a SimObject state name and plotting label with a
        user-specified name. The purpose of this register is for ease of access to SimObject
        states without having to use the satte index number.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- name - user-specified name of state
        -- id - state index number
        -- label - (optional) string other than "name" that will be displayed
                    in plots / visualization
        -------------------------------------------------------------------
        """
        self.register.update({name: {"id": id, "label": label}})


    def init_state(self, state: np.ndarray|list) -> None:
        """This method takes a numpy array or list (or nested list) and stores this data
        into the initial state SimObject.X0. This method is for user convenience when initializing
        a dynamics model.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- state - array, list or nested list of initial state array
        -------------------------------------------------------------------
        """

        if isinstance(state, list):
            state = flatten_list(state)
        _X0 = np.asarray(state).flatten()

        if _X0.shape != self.X0.shape:
            raise Exception("attempting to initialize state X0 but array sizes do not match.")

        self.X0 = _X0


    def _output_data(self, y) -> None:
        """stores solution data from solver into sim object"""
        self.Y = y


    def get_plot_data(self, subplot_id: int, index: int) -> tuple[np.ndarray, np.ndarray]:
        """This method returns state data from the SimObject according
        to the user specified state_select."""

        if not self.plot.get_config():
            Warning(f"No state_select configuration set for SimObject \"{self.name}\".")
            return (np.array([]), np.array([]))

        config_key = list(self.plot.get_config())[subplot_id]
        return (self._get_data(index, self.plot.get_config()[config_key]["xaxis"]),
                self._get_data(index, self.plot.get_config()[config_key]["yaxis"]))


    def _get_data(self, index: Optional[int], state_slice: tuple[int, int]|int|str) -> np.ndarray:
        """This method returns state data from the SimObject."""

        if index is None:
            index = len(self.Y) - 1
        else:
            index += 1 # instead of grabbin "up-to" index, grab last index as well

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
                raise Exception(f"SimObject \"{self.name}\" attempting to access state_selection \"{state_slice}\"\
                        but no state index is registered under this name.")
        else:
            return np.array([])
            

    def _set_T_array_ref(self, _T) -> None:
        """This method is used to reference the internal __T time array to the 
        Sim class Time array 'T'. This method exists to avoid redundant time arrays in
        various SimObjects."""

        self.__T = _T


    def _update_patch_data(self, xdata: np.ndarray, ydata: np.ndarray, subplot_id: int, **kwargs) -> None:
        self.plot._update_patch_data(xdata, ydata, subplot_id=subplot_id, **kwargs)


