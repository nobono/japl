import japl
import re
import numpy as np
from collections.abc import Generator
from typing import Optional, Callable
from japl.Model.Model import Model
from japl.Util.Util import flatten_list
from pyqtgraph import ScatterPlotItem, PlotDataItem, mkPen
from pyqtgraph.Qt.QtGui import QPen
from matplotlib.lines import Line2D
from matplotlib import colors as mplcolors
from pandas import DataFrame
from pandas import MultiIndex
from japl.Util.Pubsub import Publisher
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
        self.traces: list[Line2D] = []  # line / scatter plot-item for simobj (Matplotlib)
        self.qt_traces: list[PlotDataItem] = []  # line / scatter plot-items for simobj (PyQtGraph)
        self.qt_markers: list[ScatterPlotItem] = []  # marker plot-item for simobj (PyQtGraph)


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
        pen_color = self.qt_traces[subplot_id].opts['pen'].color().getRgb()[:3]  # type:ignore
        pen_width = self.qt_traces[subplot_id].opts['pen'].width()  # type:ignore
        return mkPen(pen_color, width=pen_width)


    def _update_patch_data(self, xdata: np.ndarray, ydata: np.ndarray, subplot_id: int, **kwargs) -> None:
        if (len(self.qt_traces) - 1) < subplot_id:
            return

        # update trace data
        if self.plotting_backend == "matplotlib":
            self.traces[subplot_id].set_data(xdata, ydata)

        if self.plotting_backend == "pyqtgraph":
            self.qt_traces[subplot_id].setData(x=xdata, y=ydata, **kwargs)
            if subplot_id < len(self.qt_markers):
                self.qt_markers[subplot_id].setData(x=[xdata[-1]], y=[ydata[-1]])


class SimObject:
    __slots__ = ("_dtype", "name", "color", "size", "model",
                 "state_dim", "input_dim", "static_dim",
                 "X0", "U0", "S0", "Y", "U", "plot",
                 "_T", "_istep", "publisher")

    """This is a base class for simulation objects"""



    def __init__(self, model: Model = Model(), **kwargs) -> None:

        assert isinstance(model, Model)
        self._dtype = kwargs.get("dtype", float)
        self.name = kwargs.get("name", "SimObject")
        self.color = kwargs.get("color")
        self.size = kwargs.get("size", 1)
        self.model = model
        self.state_dim = self.model.state_dim
        self.input_dim = self.model.input_dim
        self.static_dim = self.model.static_dim
        self.X0 = np.zeros((self.state_dim,))
        self.U0 = np.zeros((self.input_dim,))
        self.S0 = np.zeros((self.static_dim,))
        self.Y = np.array([], dtype=self._dtype)
        self.U = np.array([], dtype=self._dtype)
        self._T = np.array([])
        self._istep: int = 1  # sim step counter set by Sim class
        self.publisher = Publisher()

        # self._setup_model(**kwargs)

        # interface for visualization
        self.plot = _PlotInterface(
                state_select={},
                size=self.size,
                color=self.color
                )
        if not self.color:
            self.color = self.plot.color


    def _set_sim_step(self, istep: int):
        self._istep = istep


    def set_draw(self, size: float = 1, color: str = "black") -> None:
        self.size = size
        self.color = color


    # @DeprecationWarning
    # def _setup_model(self, **kwargs) -> None:
    #     # mass properties
    #     self.mass: float = kwargs.get("mass", 1)
    #     self.Ixx: float = kwargs.get("Ixx", 1)
    #     self.Iyy: float = kwargs.get("Iyy", 1)
    #     self.Izz: float = kwargs.get("Izz", 1)
    #     self.cg: float = kwargs.get("cg", 0)


    def __getattr__(self, name) -> np.ndarray|float:
        return self.get_current(name)


    def __setattr__(self, name, val) -> None:
        if name in self.__slots__:
            super().__setattr__(name, val)
        else:
            return self.set(name, val)


    def get_current(self, var_names: str|list[str]) -> np.ndarray|float:
        """This method will get data from SimObject.Y array corresponding
        to the state-name \"var_names\". but returns the current time step
        of specific variable name in the running simulation.
        """
        ret = self.get(var_names)
        if hasattr(ret, "shape"):
            if len(ret.shape) > 1:
                return ret[self._istep, :]
            elif len(ret.shape) == 1:
                return ret[self._istep]
            else:
                return ret
        else:
            return ret


    def get(self, var_names: str|list[str]) -> np.ndarray:
        """This method will get data from SimObject.Y array corresponding
        to the state-name \"var_names\".

        This method is more general, using extra checks, making is slower
        than useing get_state_array, get_input_array, or get_static_array."""

        # allow multiple names in a single string (e.g. "a, b, c")
        if isinstance(var_names, str):
            var_names = re.split(r"[,\s]", var_names)

        if len(var_names) > 1:
            ret = []
            for var_name in var_names:
                if var_name:  # string parsing may
                    if var_name in self.model.state_register:
                        ret += [self.Y[:, self.model.get_state_id(var_name)]]
                    elif var_name in self.model.input_register:
                        ret += [self.U[:, self.model.get_input_id(var_name)]]
                    elif var_name in self.model.static_register:
                        ret += [self.S0[self.model.get_static_id(var_name)]]
                    else:
                        raise Exception(f"SimObject: {self.name} cannot get model variable "
                                        f"\"{var_names}\". variable not found.")
            return np.asarray(ret).T
        else:
            if var_names[0] in self.model.state_register:
                return self.Y[:, self.model.get_state_id(var_names[0])]
            elif var_names[0] in self.model.input_register:
                return self.U[:, self.model.get_input_id(var_names[0])]
            elif var_names[0] in self.model.static_register:
                return self.S0[self.model.get_static_id(var_names[0])]
            else:
                raise Exception(f"SimObject: {self.name} cannot get model variable "
                                f"\"{var_names}\". variable not found.")


    def set(self, var_names: str|list[str], vals: float|list|np.ndarray) -> None:
        """This method will set data from SimObject.Y array corresponding
        to the state-name \"var_names\" and the current Sim time step.

        This method is more general, using extra checks, making is slower
        than useing set_state_array, set_input_array, or set_static_array."""

        # allow multiple names in a single string (e.g. "a, b, c")
        if isinstance(var_names, str) and (',' in var_names)\
                and ((names_list := var_names.split(',')).__len__() > 1):
            var_names = [i.strip() for i in names_list]

        if isinstance(var_names, list):
            for var_name in var_names:
                if var_name in self.model.state_register:
                    self.set_state_array(self.Y[self._istep], var_name, vals)
                elif var_name in self.model.input_register:
                    self.set_input_array(self.U[self._istep], var_name, vals)
                elif var_name in self.model.static_register:
                    self.set_static_array(self.S0, var_name, vals)
                else:
                    raise Exception(f"SimObject: {self.name} cannot set model variable "
                                    f"\"{var_names}\". variable not found.")
        elif isinstance(var_names, str):  # type:ignore
            if var_names in self.model.state_register:
                self.set_state_array(self.Y[self._istep], var_names, vals)
            elif var_names in self.model.input_register:
                self.set_input_array(self.U[self._istep], var_names, vals)
            elif var_names in self.model.static_register:
                self.set_static_array(self.S0, var_names, vals)
            else:
                raise Exception(f"SimObject: {self.name} cannot get model variable "
                                f"\"{var_names}\". variable not found.")
        else:
            raise Exception("unhandled case.")


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


    def get_static_array(self, static: np.ndarray, names: str|list[str]) -> np.ndarray:
        """This method gets values from the static array given the state
        names."""
        ret = self.model.get_static_id(names)
        if isinstance(names, list):
            if len(names) == 1:
                return static[ret][0]
            else:
                return static[ret]
        else:
            return static[ret]


    def set_static_array(self, static: np.ndarray, names: str|list[str],
                         vals: float|list|np.ndarray) -> None:
        """This method sets values of the static array according to the
        provided state names and provided values."""
        ret = self.model.get_static_id(names)
        if isinstance(names, list):
            if len(names) == 1:
                static[ret][0] = np.asarray(vals)
            else:
                static[ret] = np.asarray(vals)
        else:
            static[ret] = np.asarray(vals)


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
        # NOTE: ignore states in register with size > 1
        # (matrices / array) since those are there as
        # only a reference.
        state_register_num_var = len([v["var"] for v in self.model.state_register.values()
                                      if v["size"] == 1])
        input_register_num_var = len([v["var"] for v in self.model.input_register.values()
                                      if v["size"] == 1])

        if state_register_num_var != self.model.state_dim:
            raise AssertionError(f"{msg_header} state register ill-configured\n\
                                 register-dim:{len(self.model.state_register)}\n\
                                 model-state_dim:{self.model.state_dim}")

        # check inputs
        if input_register_num_var != self.model.input_dim:
            raise AssertionError(f"{msg_header} input register ill-configured\n\
                                 register-dim:{len(self.model.input_register)}\n\
                                 model-input_dim:{self.model.input_dim}")

        # TODO make sure register size and model matrix sizes agree

        # check loaded model
        assert self.model._pre_sim_checks()

        return True


    def step(self, t: float, X: np.ndarray, U: np.ndarray, S: np.ndarray, dt: float) -> np.ndarray:
        """This method is the update-step of the SimObject dynamic model. It calls
        the SimObject Model's step() function.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- t - current time
        -- X - current state array of SimObject
        -- U - current input array of SimObject
        -- S - static variables array of SimObject
        -- dt - delta time
        -------------------------------------------------------------------
        -------------------------------------------------------------------
        -- Returns:
        -------------------------------------------------------------------
        -- X_dot - state dynamics "Xdot = A*X + B*U"
        -------------------------------------------------------------------
        """
        # self.update(X)
        return self.model.step(t, X, U, S, dt)


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
            raise Exception("\n\nattempting to initialize state X0 but array sizes do not match."
                            f"\n\ninitialization array:{_X0.shape} != state array:{self.X0.shape}")
        self.X0 = _X0


    def init_static(self, state: np.ndarray|list, dtype: type = float) -> None:
        """This method takes a numpy array or list (or nested list) and stores this data
        into the initial static array SimObject.S0. This method is for user convenience when initializing
        a static variables dynamics model.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- state - array, list or nested list of static array
        -------------------------------------------------------------------
        """

        if isinstance(state, list):
            state = flatten_list(state)
        _S0 = np.asarray(state, dtype=dtype).flatten()

        if _S0.shape != self.S0.shape:
            raise Exception("\n\nattempting to initialize static array S0 but array sizes do not match."
                            f"\n\ninitialization array:{_S0.shape} != state array:{self.S0.shape}")
        self.S0 = _S0


    def get_plot_data(self, subplot_id: int, index: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
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

        if index is None or index == -1:
            index = len(self.Y) - 1
        else:
            index += 1  # instead of grabbin "up-to" index, grab last index as well

        if isinstance(state_slice, tuple) or isinstance(state_slice, list):
            return self.Y[:index, state_slice[0]:state_slice[1]]
        elif isinstance(state_slice, int):
            return self.Y[:index, state_slice]
        elif isinstance(state_slice, str):  # type:ignore
            if state_slice.lower() in ['t', 'time']:
                return self._T[:index]
            elif state_slice in self.model.state_register:
                return self.Y[:index, self.model.state_register[state_slice]["id"]]
            elif state_slice in self.model.input_register:
                return self.U[:index, self.model.input_register[state_slice]["id"]]
            else:
                raise Exception(f"SimObject \"{self.name}\" attempting to access state_selection \"{state_slice}\"\
                        but no state index is registered under this name.")
        else:
            return np.array([])


    def _set_T_array_ref(self, _T) -> None:
        """This method is used to reference the internal _T time array to the
        Sim class Time array 'T'. This method exists to avoid redundant time arrays in
        various SimObjects."""

        self._T = _T


    def _update_patch_data(self, xdata: np.ndarray, ydata: np.ndarray, subplot_id: int, **kwargs) -> None:
        self.plot._update_patch_data(xdata, ydata, subplot_id=subplot_id, **kwargs)


    def to_dataframe(self) -> DataFrame:
        """Creates DataFrame for each data array (state, input, static) on completion
        of a simulation run."""
        # define the multi-level column structure
        column_structure = []
        data = {}
        for name in self.model.state_register.keys():
            struct_tuple = ("state", name)
            data[struct_tuple] = self.get(name)
        for name in self.model.input_register.keys():
            struct_tuple = ("input", name)
            data[struct_tuple] = self.get(name)

        columns = MultiIndex.from_tuples([*data.keys()])
        df = DataFrame(data, index=self._T, columns=columns)
        return df


    def to_dict(self) -> dict:
        """Creates DataFrame for each data array (state, input, static) on completion
        of a simulation run."""
        # define the multi-level column structure
        data = {"state": {}, "input": {}}
        for name in self.model.state_register.keys():
            data["state"][name] = self.get(name)
        for name in self.model.input_register.keys():
            data["input"][name] = self.get(name)
        return data
