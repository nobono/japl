from functools import partial
import numpy as np
from typing import Callable, Optional
from typing import Generator, Union
import quaternion
from japl.Sim.Sim import Sim
from japl.SimObject.SimObject import SimObject
import pyqtgraph as pg
from pyqtgraph import GraphItem, PlotItem, mkPen
from pyqtgraph import GraphicsLayoutWidget
from pyqtgraph import PlotDataItem
from pyqtgraph import TextItem
from pyqtgraph import ViewBox
from pyqtgraph.Qt.QtGui import QColor, QKeySequence, QTransform
from pyqtgraph.Qt.QtWidgets import QApplication
from pyqtgraph.Qt.QtWidgets import QShortcut
from pyqtgraph.Qt.QtCore import QCoreApplication
from pyqtgraph.Qt.QtCore import QTimer
# from japl.Math.Rotation import quat_to_tait_bryan
# from pyqtgraph.Qt.QtWidgets import QGridLayout, QWidget, QWidgetItem
# from pyqtgraph import PlotItem, QtGui, mkColor, mkPen, PlotCurveItem
# from pyqtgraph import PlotWidget
# from pyqtgraph.Qt.QtCore import QRectF
from matplotlib import colors as mplcolors
from japl.Util.Profiler import Profiler



PlotObj = Union[Sim, SimObject, tuple, list, None]



class PyQtGraphPlotter:

    def __init__(self, **kwargs) -> None:
        # essentials
        self.istep = 0
        self.simobjs = []
        self.Nt = kwargs.get("Nt", 1)
        self.dt = kwargs.get("dt", None)

        # plotting
        self.figsize: tuple = kwargs.get("figsize", (6, 4))
        self.aspect: float|str = kwargs.get("aspect", "equal")
        self.blit: bool = kwargs.get("blit", False)
        self.cache_frame_data: bool = kwargs.get("cache_frame_data", False)
        self.repeat: bool = kwargs.get("repeat", False)
        self.antialias: bool = kwargs.get("antialias", True)
        self.instrument_view: bool = kwargs.get("instrument_view", False)
        self.draw_cache_mode: bool = kwargs.get("draw_cache_mode", False)
        self.frame_rate: float = kwargs.get("frame_rate", 10)
        self.moving_bounds: bool = kwargs.get("moving_bounds", False)
        self.xlim: list = kwargs.get("xlim", [])
        self.ylim: list = kwargs.get("ylim", [])
        self.ff: float = kwargs.get("ff", 1)  # fast-forward multplier

        # debug
        self.quiet = kwargs.get("quiet", False)
        self.instrument_view &= not self.quiet
        self.profiler = Profiler()

        # colors
        self.COLORS = dict(mplcolors.TABLEAU_COLORS, **mplcolors.CSS4_COLORS)  # available colors
        self.color_cycle = self.__color_cycle()  # color cycle list
        self.background_color = kwargs.get("background_color", "black")
        self.text_color = kwargs.get("text_color", "grey")

        # configure pyqtgraph options
        pg.setConfigOptions(antialias=self.antialias)

        self.wins: list[GraphicsLayoutWidget] = []  # contains view layouts for each simobj
        self.shortcuts = []
        self.timer: Optional[QTimer] = None
        self.app = self.setup()


    def setup(self) -> QCoreApplication:
        """This method starts the Qt Application."""
        # Always start by initializing Qt (only once per application)
        # if QApplication instance already running, use running instance.
        if (app := QCoreApplication.instance()):
            return app
        else:
            if self.quiet:
                return QCoreApplication([])  # no GUI
            else:
                return QApplication([])   # GUI


    def close_windows(self) -> None:
        for win in self.wins:
            win.close()


    def exit(self) -> None:
        if self.app:
            if self.quiet:
                self.app.exit()  # immediately exit
            if self.timer and self.timer.isActive():
                self.timer.stop()  # stop timer but keep windows open
            else:
                self.app.exit()


    def animate(self, plot_obj: PlotObj) -> "PyQtGraphPlotter":
        """This method sets up animation plots. The purpose of this
        method is to execute animated plots for certain provided
        argument types \"PlotObj\"."""
        self.__setup_from_plot_obj(plot_obj)

        # TODO: handle multiple simobjs
        if len(self.simobjs) < 1:
            return self

        simobj = self.simobjs[0]

        # get some values from Sim
        if isinstance(plot_obj, Sim):
            # setup simobj plots
            self.add_simobject(simobj)

            step_func = plot_obj._step_solve
            dynamics_func = plot_obj.step
            method = plot_obj.integrate_method
            rtol = plot_obj.rtol
            atol = plot_obj.atol
            dt = self.dt
            interval_ms = int(max(1, (1 / self.frame_rate) * 1000))

            # create function for each time step
            step_func = partial(step_func,
                                dynamics_func=dynamics_func,
                                istep=0,
                                dt=dt,
                                simobj=simobj,
                                method=method,
                                rtol=rtol,
                                atol=atol)

            # create function for animation of each time step
            anim_func = partial(self._animate_func,
                                simobj=simobj,
                                step_func=step_func,
                                frame_rate=interval_ms,
                                moving_bounds=self.moving_bounds)

            anim = self.FuncAnimation(  # noqa
                    func=anim_func,
                    frames=self.Nt,
                    interval=interval_ms,
                    )

        elif isinstance(plot_obj, SimObject):
            pass
        elif plot_obj.__class__ in [list, tuple, np.ndarray]:
            # NOTE:
            # assume x-value as time
            # or specify rate?

            pass
        elif plot_obj is None:
            pass
        else:
            raise Exception("cannot input this object type to Plotter.")

        return self


    def show(self) -> None:
        # first show on windows
        for win in self.wins:
            win.show()
        if self.app:
            self.app.exec_()  # or app.exec_() for PyQt5 / PySide2
        else:
            raise Exception("Trying to show() but PyQtGraphPlotter has not been setup.")

    # --------------------------------------------------------------------------------------
    # Helper Methods
    # --------------------------------------------------------------------------------------

    def __color_cycle(self) -> Generator[str, None, None]:
        """This method is a Generator which handles the color cycle of line / scatter
        plots which do not specify a color."""
        while True:
            for _, v in self.COLORS.items():
                yield str(v)


    def __get_color_code(self, color: str) -> str:
        if color in self.COLORS:
            color_code = self.COLORS[color]
            return str(color_code)
        else:
            raise Exception(f"color \"{color}\" not available.")


    def __setup_from_plot_obj(self, plot_obj: PlotObj) -> None:
        """This method initializes Plotter class from a variety of
        plottable objects (PlotObj)."""
        if isinstance(plot_obj, Sim):
            # get some values from Sim
            self.Nt = plot_obj.Nt
            self.dt = plot_obj.dt
            self.simobjs = plot_obj.simobjs
        elif isinstance(plot_obj, SimObject):
            pass
        elif isinstance(plot_obj, tuple) or isinstance(plot_obj, list):
            pass
        elif plot_obj is None:
            pass
        else:
            raise Exception("cannot input this object type to Plotter.")


    def __check_animate_stop(self, N: Callable|Generator|int) -> None:
        """This method is for animated plots and exits if the number
        of steps has reached the number of desired steps."""
        if isinstance(N, Callable):
            if not N():
                self.exit()
        elif isinstance(N, Generator):
            if self.istep >= len(list(N)):
                self.exit()
        elif isinstance(N, int):  # type:ignore
            if self.istep >= N:
                self.exit()


    def __apply_style_settings_to_plot(self, plot_item):
        plot_item.showGrid(True, True, 0.5)
        plot_item.setAspectLocked(self.aspect == "equal")

    # --------------------------------------------------------------------------------------
    # ViewBoxes
    # --------------------------------------------------------------------------------------

    def get_text_viewbox(self, window_id: int) -> Optional[ViewBox]:
        """Returns the ViewBox associated with provided window_id"""
        if self.instrument_view:
            win = self.wins[window_id]
            # TODO right now row/col for text view is static
            text_view_row = 2
            text_view_col = 1
            return win.getItem(text_view_row, text_view_col)


    def get_text_item(self, window_id: int, text_item_id: int) -> Optional[TextItem]:
        """Returns the TextItem associated with the provided window_id and text_item_id"""
        if (text_viewbox := self.get_text_viewbox(window_id)):
            text_item: TextItem = text_viewbox.addedItems[text_item_id]
            return text_item


    def add_text(self, text: str, window_id: int = 0, color: tuple = (255, 255, 255),
                 spacing: float = 0.6) -> None:
        if (text_viewbox := self.get_text_viewbox(window_id)):
            ntext = len(text_viewbox.addedItems)
            text_viewbox.addItem(TextItem(
                text,
                color=color,
                anchor=(0, 3 - spacing * ntext),
                ))


    def set_text(self, text: str, window_id: int = 0, text_item_id: int = 0) -> None:
        if (text_item := self.get_text_item(window_id, text_item_id)):
            text_item.setText(text)

    # --------------------------------------------------------------------------------------
    # Graphic Items
    # --------------------------------------------------------------------------------------

    # TODO finish this...
    def add_vector(self,
                   p0: np.ndarray,
                   p1: np.ndarray,
                   view: ViewBox,
                   rgba: tuple = (255, 255, 255, 255),
                   width: float = 2) -> None:

        vector_item: GraphItem = GraphItem()
        vector_verts = np.array([p0, p1])
        vector_conn = np.array([[0, 1]])
        vector_lines = np.array(
                [(rgba + (width,))] * len(vector_conn),
                dtype=[
                    ('red', np.ubyte),
                    ('green', np.ubyte),
                    ('blue', np.ubyte),
                    ('alpha', np.ubyte),
                    ('width', float),
                    ])
        vector_item.setData(
                pos=vector_verts,
                adj=vector_conn,
                pen=vector_lines,
                size=1,
                symbol=None,
                pxMode=False
                )
        view.addItem(vector_item)  # type:ignore

    # --------------------------------------------------------------------------------------
    # Plotter Build Methods
    # --------------------------------------------------------------------------------------

    def create_window(self, figsize: Optional[list[float]|tuple[float]] = None) -> GraphicsLayoutWidget:
        """This method adds a window popup to the Application
        instance. A keyboard shortcut 'q' is also added to close
        said window."""
        if not figsize:
            figsize = self.figsize

        # configure window
        win = GraphicsLayoutWidget()
        win.resize(*(np.array([*figsize]) * 100))

        # set background color
        win.setBackground(self.background_color)

        # shortcut keys callbacks for each simobj view
        shortcut = QShortcut(QKeySequence("Q"), win)
        shortcut.activated.connect(self.close_windows)

        # add to lists
        self.wins.append(win)
        self.shortcuts.append(shortcut)
        return win


    def add_plot_to_window(self,
                           win: GraphicsLayoutWidget|int,
                           title: str = "",
                           row: int = 0,
                           col: int = 0,
                           color_code: str = "",
                           size: float = 1,
                           aspect: str = "",
                           show_grid: bool = True,
                           xlabel: str = "",
                           ylabel: str = "") -> PlotDataItem:
        """This method adds a plot to a window. This method adds a plot
        to a specified window; then adds a PlotDataItem to the newly created
        PlotItem."""
        # if "win" is window-id, get from list
        if isinstance(win, int):
            win = self.wins[win]

        pen = {"color": color_code, "width": size}
        plot_item = win.addPlot(row=row, col=col, colspan=2, title=title, name=title)
        if self.xlim:
            plot_item.setRange(xRange=self.xlim)
        if self.ylim:
            plot_item.setRange(yRange=self.ylim)
        # enable autorange
        # plot_item.enableAutoRange(axis='xy', enable=True)
        self.__apply_style_settings_to_plot(plot_item)

        # style settings
        text_color_code = self.__get_color_code(self.text_color)
        plot_item.setTitle(title, color=self.text_color)
        plot_item.setLabel("bottom", xlabel, color=self.text_color)
        plot_item.setLabel("left", ylabel, color=self.text_color)
        plot_item.getAxis("left").setPen({"color": text_color_code})
        plot_item.getAxis("bottom").setPen({"color": text_color_code})
        graphic_item = PlotDataItem(x=[],
                                    y=[],
                                    pen=pen,
                                    useCache=self.draw_cache_mode,
                                    antialias=self.antialias,
                                    autoDownsample=True,
                                    downsampleMethod="peak",
                                    clipToView=True,
                                    skipFiniteCheck=True)
        plot_item.addItem(graphic_item)   # init PlotCurve
        return graphic_item


    def add_simobject(self, simobj: SimObject) -> None:
        if self.quiet:
            return

        self.simobjs += [simobj]
        win = self.create_window()
        num_plots = 0

        # add plot-style to each window using SimObject._PlotInterface
        # config dict.
        for i, (title, axes) in enumerate(simobj.plot.get_config().items()):
            # get values from plot config
            # if no color specified in config,
            # try to get color from simobj
            aspect = axes.get("aspect", self.aspect)
            color = axes.get("color", simobj.plot.color)
            size = axes.get("size", 1)
            marker = axes.get("marker", None)

            # resolve color str to hex color code
            # or get random color
            if color:
                color_code = self.__get_color_code(color)
            else:
                color_code = next(self.color_cycle)

            # setup line plot for simobj
            graphic_item = self.add_plot_to_window(win=win, title=title, row=i, col=0,
                                                   color_code=color_code, size=size, aspect=aspect,
                                                   plot_item_type=pg.PlotDataItem)
            simobj.plot.qt_traces += [graphic_item]

            # setup marker for simobj
            if marker:
                marker_pen = {"color": color_code, "width": size}
                marker_item = pg.ScatterPlotItem(x=[], y=[], pen=marker_pen,
                                                 useCache=self.draw_cache_mode,
                                                 antialias=self.antialias,
                                                 autoDownsample=True,
                                                 downsampleMethod="peak",
                                                 clipToView=True,
                                                 skipFiniteCheck=True,
                                                 symbol=marker)
                # refer to current plot in current window
                plot_item = win.getItem(row=i, col=0)
                plot_item.addItem(marker_item)
                simobj.plot.qt_markers += [marker_item]
            num_plots += 1

        # setup vehicle viewer widget
        if self.instrument_view:
            _view = ViewBox(name="instrument_view")
            _view.setAspectLocked(True)
            _view.setRange(xRange=[-1, 1], yRange=[-1, 1])
            win.addItem(_view, row=(num_plots + 1), col=0, colspan=1)  # type:ignore

            # ViewBox for text
            _text_view = ViewBox()
            _text_view.setRange(xRange=[-1, 1], yRange=[-1, 1])
            win.addItem(_text_view, row=(num_plots + 1), col=1, colspan=1)  # type:ignore

            self.attitude_graph_item: GraphItem = GraphItem()
            _view.addItem(self.attitude_graph_item)  # type:ignore

            # missile drawing
            L = .5
            H = .05
            nose_len = 0.1
            cx, cy = (0, 0)

            # Define positions of nodes
            self.attitude_graph_verts = np.array([
                [cx, cy],
                [cx - L, cy - H],
                [cx - L, cy + H],
                [cx + L, cy + H],
                [cx + L, cy - H],
                [cx + L + nose_len, cy]
            ])

            # Define the set of connections in the graph
            self.attitude_graph_conn = np.array([
                [1, 2],
                [2, 3],
                [3, 5],
                [5, 4],
                [4, 1],
            ])

            # Define the symbol to use for each node (this is optional)
            # self.symbols = ['x', 'x', 'x', 'x', 'x', 'x']
            self.symbols = None

            # Define the line style for each connection (this is optional)
            self.attitude_graph_lines = np.array(
                    [(100, 100, 255, 255, 8)] * len(self.attitude_graph_conn),
                    dtype=[
                        ('red', np.ubyte),
                        ('green', np.ubyte),
                        ('blue', np.ubyte),
                        ('alpha', np.ubyte),
                        ('width', float),
                        ])

            # Update the graph
            self.attitude_graph_item.setData(
                    pos=self.attitude_graph_verts,
                    adj=self.attitude_graph_conn,
                    pen=self.attitude_graph_lines,
                    size=1,
                    symbol=self.symbols,
                    pxMode=False
                    )

    # --------------------------------------------------------------------------------------
    # Animation
    # --------------------------------------------------------------------------------------

    def FuncAnimation(self,
                      func: Callable,
                      frames: Callable|Generator|int,
                      interval: int,
                      ) -> None:
        # plotter.widget
        self.timer = QTimer()
        self.timer.timeout.connect(partial(func, frame=0))
        self.timer.timeout.connect(partial(self.__check_animate_stop, frames))
        self.timer.start(interval)


    def _animate_func(self, frame, simobj: SimObject, step_func: Callable,
                      frame_rate: float, moving_bounds: bool = False):

        self.profiler()

        # run ODE solver step
        nsteps = max(1, int(frame_rate / (self.dt * 1000)))
        for _ in range(int(nsteps * self.ff)):
            self.istep += 1
            if self.istep <= self.Nt:
                flag_sim_stop: bool = step_func(istep=self.istep)
                if flag_sim_stop:
                    self.exit()
            else:
                break

        # for subplot_id in range(len(simobj.plot.get_config())):
        plot_config = simobj.plot.get_config()
        for subplot_id, plot_name in enumerate(plot_config):
            # get data from SimObject based on state_select user configuration
            # NOTE: can pass QPen to _update_patch_data
            xdata, ydata = simobj.get_plot_data(subplot_id, self.istep)
            simobj._update_patch_data(xdata, ydata, subplot_id=subplot_id)

            ##########################################
            # this code allows xlim, ylim ranges to be
            # defined and used until data goes outside
            # of bounds and autorange is enabled.
            ##########################################
            widget_items = list(self.wins[0].centralWidget.items.keys())  # type:ignore
            subplot = plot_config[plot_name]
            # xrange, yrange = widget_items[subplot_id].viewRange()
            xlim = subplot.get("xlim", None)
            ylim = subplot.get("ylim", None)
            if xlim:
                if (xdata[-1] < xlim[0]) or (xdata[-1] > xlim[1]):
                    widget_items[subplot_id].enableAutoRange(x=True)
                else:
                    widget_items[subplot_id].setRange(xRange=xlim)
            if ylim:
                if (ydata[-1] < ylim[0]) or (ydata[-1] > ylim[1]):
                    widget_items[subplot_id].enableAutoRange(y=True)
                else:
                    widget_items[subplot_id].setRange(yRange=ylim)

        # drawing the instrument view of vehicle
        # TODO generalize: each simobj has its own body to draw.
        if self.instrument_view:
            self._draw_instrument_view(simobj)


    def _draw_instrument_view(self, _simobj: SimObject) -> None:
        """This method updates the instrument ViewBox.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- _simobj - (SimObject)
        -------------------------------------------------------------------
        """

        ####################################################################
        # NOTE:
        # pyqtgraph coordinate convention is:
        #   - X: points to the right
        #   - Y: points upward
        #   - Z: points outward
        #
        # when applying a rotation matrix, a permutation matrix is required
        # to meet this convention (swapping rows 1 & 2). When applying a
        # quaternion the 'y' & 'z' (q2 & q3) components must be swapped.
        ####################################################################

        # get quaternion from current state
        istate = _simobj.model.get_current_state()
        iquat = _simobj.get_state_array(istate, ["q_0", "q_1", "q_2", "q_3"])

        # get rotation matrix
        _iquat = quaternion.from_float_array(iquat)
        dcm = quaternion.as_rotation_matrix(_iquat).flatten()

        # swap rows 1 & 2; swap cols 1 & 2
        yz_swapped_dcm = np.array([dcm[0], dcm[2], dcm[1],
                                   dcm[6], dcm[8], dcm[7],
                                   dcm[3], dcm[5], dcm[4]])

        transform = QTransform(*yz_swapped_dcm)
        self.attitude_graph_item.setTransform(transform)


    # def _time_slider_update(self, val: float, _simobjs: list[SimObject]) -> None:
    #     pass


    # def set_lim(self, lim: list|tuple, padding=0.02) -> None:
    #     assert len(lim) == 4

    #     x = lim[0]
    #     y = lim[2]
    #     width = lim[1] - lim[0]
    #     height = lim[3] - lim[2]

    #     newRect = QRectF(x, y, width, height)  # type:ignore
    #     self.widget.setRange(newRect, padding=padding)


    # def __x_axis_right_border_append(self, ax: Axes, val: float):
    #     pass


    # def __x_axis_left_border_append(self, ax: Axes, val: float):
    #     pass


    # def __y_axis_top_border_append(self, ax: Axes, val: float):
    #     pass


    # def __y_axis_bottom_border_append(self, ax: Axes, val: float):
    #     pass


    # def update_axes_boundary(self, ax: Axes, pos: list|tuple, margin: float = 0.1,
    #                          moving_bounds: bool = False) -> None:
    #     """
    #         This method handles the plot axes boundaries during FuncAnimation frames.

    #     -------------------------------------------------------------------
    #     -- Arguments
    #     -------------------------------------------------------------------
    #     -- ax - matplotlib Axes object
    #     -- pos - xy position of Artist being plotted
    #     -- margin - % margin value between Artist xy position and Axes border
    #     -------------------------------------------------------------------
    #     """

    #     xlim = ax.get_xlim()
    #     ylim = ax.get_ylim()
    #     xcenter, ycenter = pos

    #     xlen = xlim[1] - xlim[0]
    #     ylen = ylim[1] - ylim[0]

    #     aspect_ratio = 1.4
    #     X_RANGE_LIM = 100
    #     Y_RANGE_LIM = int(X_RANGE_LIM / aspect_ratio)

    #     # weight the amount to move the border proportional to how close
    #     # the object cetner is to the border limit. Also account for the
    #     # scale of the current plot window (xlen, ylen) in the amount to
    #     # change the current boundary.

    #     if (weight := xcenter - xlim[1]) + margin > 0:
    #         length_weight = min(xlen, 20)
    #         self.__x_axis_right_border_append(ax, margin * length_weight * abs(weight))
    #         if xlen > X_RANGE_LIM and moving_bounds:
    #             self.__x_axis_left_border_append(ax, margin * length_weight * abs(weight))

    #     if (weight := xcenter - xlim[0]) - margin < 0:
    #         length_weight = min(xlen, 20)
    #         self.__x_axis_left_border_append(ax, -(margin * length_weight * abs(weight)))
    #         if xlen > X_RANGE_LIM and moving_bounds:
    #             self.__x_axis_right_border_append(ax, -(margin * length_weight * abs(weight)))

    #     if (weight := ycenter - ylim[1]) + margin > 0:
    #         length_weight = min(ylen, 20)
    #         self.__y_axis_top_border_append(ax, margin * length_weight * abs(weight))
    #         if ylen > Y_RANGE_LIM and moving_bounds:
    #             self.__y_axis_bottom_border_append(ax, margin * length_weight * abs(weight))

    #     if (weight := ycenter - ylim[0]) - margin < 0:
    #         length_weight = min(ylen, 20)
    #         self.__y_axis_bottom_border_append(ax, -(margin * length_weight * abs(weight)))
    #         if ylen > Y_RANGE_LIM and moving_bounds:
    #             self.__y_axis_top_border_append(ax, -(margin * length_weight * abs(weight)))


    # def setup_time_slider(self, Nt: int, _simobjs: list[SimObject]) -> None:
    #     pass


    def plot_obj(self, simobj: SimObject, **kwargs) -> "PyQtGraphPlotter":
        self.add_simobject(simobj)
        for subplot_id in range(len(simobj.plot.get_config())):
            # get data from SimObject based on state_select user configuration
            # NOTE: can pass QPen to _update_patch_data
            xdata, ydata = simobj.get_plot_data(subplot_id, index=-1)
            simobj._update_patch_data(xdata, ydata, subplot_id=subplot_id)
        return self


    def figure(self):
        self.create_window()


    def plot(self,
             x: np.ndarray|list,
             y: np.ndarray|list,
             color: str = "",
             size: float = 2,
             marker: Optional[str] = None,
             marker_size: int = 1,
             window_id: int = 0,
             title: str = "",
             xlabel: str = "",
             ylabel: str = "",
             legend_name: str = "",
             **kwargs) -> PlotItem:

        # convert mpl color to rgb
        if color:
            color_code = self.__get_color_code(color)
        else:
            color_code = next(self.color_cycle)

        # data = {'x': x, 'y': y}
        pen = {"color": color_code, "width": size}
        symbol_pen = {"color": color_code, "width": marker_size}

        # old
        ###########################################################################
        # line = pg.PlotCurveItem(x=x, y=y, pen=pen, symbol=marker)
        # graphic_item = self.add_plot_to_window(win=win, title=title, row=0, col=0,
        #                                        color=color, size=size)
        # graphic_item.setData(data, symbol=marker, symbolPen=symbol_pen, **kwargs)
        ###########################################################################

        # check if at least 1 window avaiable
        # then get GraphicsItem at [0, 0] (default)
        if len(self.wins) < 1:
            win = self.create_window()
            plot_item: PlotItem = win.addPlot(row=0, col=0, title=title, name=title)
            text_color_code = self.__get_color_code(self.text_color)
            legend = plot_item.addLegend()
            legend.setBrush('k')
            legend.setPen({"color": self.__get_color_code("black")})
            # style settings
            plot_item.setTitle(title, color=self.text_color)
            plot_item.setLabel("bottom", xlabel, color=self.text_color)
            plot_item.setLabel("left", ylabel, color=self.text_color)
            plot_item.getAxis("left").setPen({"color": text_color_code})
            plot_item.getAxis("bottom").setPen({"color": text_color_code})
        else:
            win = self.wins[-1]  # get last created window
            plot_item = win.getItem(row=0, col=0)
            if plot_item is None:
                plot_item: PlotItem = win.addPlot(row=0, col=0, title=title, name=title)
                legend = plot_item.addLegend()
                legend.setBrush('k')
                legend.setPen({"color": self.__get_color_code("black")})
                # style settings
                text_color_code = self.__get_color_code(self.text_color)
                plot_item.setTitle(title, color=self.text_color)
                plot_item.setLabel("bottom", xlabel, color=self.text_color)
                plot_item.setLabel("left", ylabel, color=self.text_color)
                plot_item.getAxis("left").setPen({"color": text_color_code})
                plot_item.getAxis("bottom").setPen({"color": text_color_code})
        scatter = pg.PlotCurveItem(x=x, y=y, pen=pen, symbol=marker, symbolPen=symbol_pen)
        plot_item.legend.addItem(scatter, legend_name)  # type:ignore
        plot_item.addItem(scatter)
        self.__apply_style_settings_to_plot(plot_item)
        return plot_item


    def scatter(self,
                x: np.ndarray|list,
                y: np.ndarray|list,
                color: str = "",
                size: int = 2,
                marker: str = "o",
                window_id: int = 0,
                title: str = "",
                xlabel: str = "",
                ylabel: str = "",
                **kwargs) -> PlotItem:

        # convert mpl color to rgb
        if color:
            color_code = self.__get_color_code(color)
        else:
            color_code = next(self.color_cycle)

        pen = {"color": color_code, "width": size}
        symbol_pen = {"color": color_code, "width": size}

        # if someone sets empty marker
        if not marker:
            marker = 'o'

        # check if at least 1 window avaiable
        # then get GraphicsItem at [0, 0] (default)
        if len(self.wins) < 1:
            win = self.create_window()
            plot_item = win.addPlot(row=0, col=0, title=title, name=title)
            # style settings
            text_color_code = self.__get_color_code(self.text_color)
            plot_item.setTitle(title, color=self.text_color)
            plot_item.setLabel("bottom", xlabel, color=self.text_color)
            plot_item.setLabel("left", ylabel, color=self.text_color)
            plot_item.getAxis("left").setPen({"color": text_color_code})
            plot_item.getAxis("bottom").setPen({"color": text_color_code})
        else:
            win = self.wins[-1]
            plot_item = win.getItem(row=0, col=0)
            # style settings
            text_color_code = self.__get_color_code(self.text_color)
            plot_item.setTitle(title, color=self.text_color)
            plot_item.setLabel("bottom", xlabel, color=self.text_color)
            plot_item.setLabel("left", ylabel, color=self.text_color)
            plot_item.getAxis("left").setPen({"color": text_color_code})
            plot_item.getAxis("bottom").setPen({"color": text_color_code})

        scatter = pg.ScatterPlotItem(x=x, y=y, pen=pen, symbol=marker, symbolPen=symbol_pen)
        plot_item.addItem(scatter)
        self.__apply_style_settings_to_plot(plot_item)
        return plot_item


    # def __handle_plot_creation(self):
    #     """
    #     - adding plot items should be applied to a
    #         single window unless otherswise specified.
    #     - adding of plot items should be applied to
    #         row=0, col=0 unless otherwise specified.
    #     """
    #     pass


    # def autoscale(self, xdata: np.ndarray|list, ydata: np.ndarray|list) -> None:
    #     # autoscale
    #     self.set_lim([min(xdata) - 0.2, max(xdata) + 0.2, min(ydata) - 0.2, max(ydata) + 0.2])
