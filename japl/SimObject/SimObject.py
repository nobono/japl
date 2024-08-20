import japl
import numpy as np
from collections.abc import Generator
from typing import Optional
from japl.Aero.AeroTable import AeroTable
from japl.Model.Model import Model
from japl.Util.Util import flatten_list
from pyqtgraph import PlotDataItem, mkPen
from pyqtgraph.Qt.QtGui import QPen
from matplotlib.lines import Line2D
from matplotlib import colors as mplcolors
# from sympy import Symbol
# from pyqtgraph import GraphicsView, PlotCurveItem,
# from pyqtgraph import CircleROI
# from matplotlib.axes import Axes
# import astropy.units as u
# from astropy.units.quantity import Quantity
# from japl.Util.UnitCheck import assert_physical_type
# from japl.Model.Model import ModelType
# from matplotlib.patches import Circle



# class ShapeCollection:
#
#     """This is a class which abstracts the line / shape plots of different
#     plotting backends."""

#     def __init__(self, color: str, radius: float) -> None:
#         # assert plotting_backend in ["matplotlib", "pyqtgraph", "mpl", "qt"]
#         # self.plotting_backend = plotting_backend
#         self.color = color
#         self.radius = radius


#     def setup(self):
#         self.patch = Circle((0, 0), radius=size, color=color)
#         self.trace = Line2D([0], [0], color=color)



class _PlotInterface:

    """This is a class for interfacing SimObject data with the plotter."""

    def __init__(self, size: float, state_select: dict, color: Optional[str] = None) -> None:

        # available colors
        self.COLORS = dict(mplcolors.TABLEAU_COLORS, **mplcolors.CSS4_COLORS)

        self.size = size

        # color cycle list
        self.color_cycle = self.__color_cycle()
        self.__plot_config = state_select
        self.plotting_backend = japl.get_plotlib()

        if not color:
            self.color_code = next(self.color_cycle)
            self.color = list(self.COLORS.keys())[
                    list(self.COLORS.values()).index(self.color_code)]
        else:
            self.color = color
            self.color_code = self.get_mpl_color_code(self.color)

        # graphic objects
        self.traces: list[Line2D] = []
        self.qt_traces: list[PlotDataItem] = []


    def set_config(self, plot_config: dict) -> None:
        # TODO check format here
        self.__plot_config = plot_config


    def get_config(self) -> dict:
        return self.__plot_config


    def get_mpl_color_code(self, color_str: str = "") -> str:
        color_code = self.COLORS[color_str]
        return str(color_code)


    def __color_cycle(self) -> Generator[str, None, None]:
        """This method is a Generator which handles the color cycle of line / scatter
        plots which do not specify a color."""

        while True:
            for _, v in self.COLORS.items():
                yield str(v)


    @DeprecationWarning
    def _get_qt_pen(self, subplot_id: int) -> QPen:
        pen_color = self.qt_traces[subplot_id].opts['pen'].color().getRgb()[:3]
        pen_width = self.qt_traces[subplot_id].opts['pen'].width()
        return mkPen(pen_color, width=pen_width)


    def _update_patch_data(self, xdata: np.ndarray, ydata: np.ndarray, subplot_id: int, **kwargs) -> None:
        if (len(self.qt_traces) - 1) < subplot_id:
            return

        # update trace data
        if self.plotting_backend == "matplotlib":
            self.traces[subplot_id].set_data(xdata, ydata)

        if self.plotting_backend == "pyqtgraph":
            self.qt_traces[subplot_id].setData(x=xdata, y=ydata, **kwargs)


class SimObject:

    """This is a base class for simulation objects"""

    def __init__(self, model: Model = Model(), **kwargs) -> None:

        assert isinstance(model, Model)
        self._dtype = kwargs.get("dtype", float)
        self.name = kwargs.get("name", "SimObject")
        self.color = kwargs.get("color")
        self.size = kwargs.get("size", 1)
        self.model = model
        self.state_dim = self.model.state_dim
        self.X0 = np.zeros((self.state_dim,))
        self.Y = np.array([], dtype=self._dtype)
        self.__T = np.array([])

        self._setup_model(**kwargs)

        self.aerotable: Optional[AeroTable] = kwargs.get("aerotable", None)

        # interface for visualization
        self.plot = _PlotInterface(
                state_select={},
                size=self.size,
                color=self.color
                )
        if not self.color:
            self.color = self.plot.color


    def set_draw(self, size: float = 1, color: str = "black") -> None:
        self.size = size
        self.color = color


    def _setup_model(self, **kwargs) -> None:
        # mass properties
        self.mass: float = kwargs.get("mass", 1)
        self.Ixx: float = kwargs.get("Ixx", 1)
        self.Iyy: float = kwargs.get("Iyy", 1)
        self.Izz: float = kwargs.get("Izz", 1)
        self.cg: float = kwargs.get("cg", 0)


    def get_state_array(self, state: np.ndarray, names: str|list[str]) -> np.ndarray:
        """This method gets values from the state array given the state
        names."""
        ret = self.model.get_state_id(names)
        if isinstance(names, list):
            if len(names) == 1:
                return state[ret][0]
            else:
                return state[ret]
        else:
            return state[ret]


    def set_state_array(self, state: np.ndarray, names: str|list[str],
                        vals: float|list|np.ndarray) -> None:
        """This method sets values of the state array according to the
        provided state names and provided values."""
        ret = self.model.get_state_id(names)
        if isinstance(names, list):
            if len(names) == 1:
                state[ret][0] = np.asarray(vals)
            else:
                state[ret] = np.asarray(vals)
        else:
            state[ret] = np.asarray(vals)


    def get_input_array(self, input: np.ndarray, names: str|list[str]) -> float|np.ndarray:
        """This method gets values from the input array given the input
        names."""
        ret = self.model.get_input_id(names)
        if isinstance(names, list):
            if len(names) == 1:
                return input[ret][0]
            else:
                return input[ret]
        else:
            return input[ret]


    def set_input_array(self, input: np.ndarray, names: str|list[str],
                        vals: float|list|np.ndarray) -> None:
        """This method sets values of the input array according to the
        provided input names and provided values."""
        ret = self.model.get_input_id(names)
        if isinstance(names, list):
            if len(names) == 1:
                input[ret][0] = np.asarray(vals)
            else:
                input[ret] = np.asarray(vals)
        else:
            input[ret] = np.asarray(vals)


    def _pre_sim_checks(self) -> bool:
        """This method is used to check user configuration of SimObjects
        before running a simulation."""

        msg_header = f"SimObject \"{self.name}\""

        # check if model exists
        if self.model is None:
            raise AssertionError(f"{msg_header} has no model")

        # check initial state aray
        if len(self.X0) != self.model.state_dim:
            raise AssertionError(f"{msg_header} initial state \"X0\" ill-configured")

        # check state / output register
        if len(self.model.state_register) != self.model.state_dim:
            raise AssertionError(f"{msg_header} state register ill-configured\n\
                                 register-dim:{len(self.model.state_register)}\n\
                                 model-state_dim:{self.model.state_dim}")

        # check inputs
        if len(self.model.input_register) != self.model.input_dim:
            raise AssertionError(f"{msg_header} input register ill-configured\n\
                                 register-dim:{len(self.model.input_register)}\n\
                                 model-input_dim:{self.model.input_dim}")

        # TODO make sure register size and model matrix sizes agree

        # check loaded model
        assert self.model._pre_sim_checks()

        return True


    def step(self, X: np.ndarray, U: np.ndarray, dt: float) -> np.ndarray:
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
        self.update(X)
        return self.model.step(X, U, dt)


    def update(self, X: np.ndarray):
        pass


    def init_state(self, state: np.ndarray|list, dtype: type = float) -> None:
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
        _X0 = np.asarray(state, dtype=dtype).flatten()

        if _X0.shape != self.X0.shape:
            raise Exception("attempting to initialize state X0 but array sizes do not match.")

        self.X0 = _X0


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
            index += 1  # instead of grabbin "up-to" index, grab last index as well

        if isinstance(state_slice, tuple) or isinstance(state_slice, list):
            return self.Y[:index, state_slice[0]:state_slice[1]]
        elif isinstance(state_slice, int):
            return self.Y[:index, state_slice]
        elif isinstance(state_slice, str):
            if state_slice.lower() in ['t', 'time']:
                return self.__T[:index]
            elif state_slice in self.model.state_register:
                return self.Y[:index, self.model.state_register[state_slice]["id"]]
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
