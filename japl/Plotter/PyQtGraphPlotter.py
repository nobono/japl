from functools import partial
import numpy as np

from typing import Callable, Optional
from typing import Generator

from pyqtgraph.Qt.QtWidgets import QGridLayout, QWidget, QWidgetItem
import quaternion


from japl.Math.Rotation import quat_to_tait_bryan
from japl.SimObject.SimObject import SimObject

import pyqtgraph as pg
from pyqtgraph import GraphItem, GraphicsLayoutWidget, PlotCurveItem, PlotDataItem, PlotItem, QtGui, TextItem, ViewBox, mkColor, mkPen
from pyqtgraph import QtWidgets
from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt.QtCore import QRectF
from pyqtgraph.Qt.QtGui import QKeySequence, QTransform

from matplotlib import colors as mplcolors

import time
# ---------------------------------------------------



class PyQtGraphPlotter:

    def __init__(self, Nt: int, **kwargs) -> None:

        self.Nt = Nt
        self.simobjs = []
        self.dt = kwargs.get("dt", None)

        # plotting
        self.figsize = kwargs.get("figsize", (6, 4))
        self.aspect: float|str = kwargs.get("aspect", "equal")
        self.blit: bool = kwargs.get("blit", False)
        self.cache_frame_data: bool = kwargs.get("cache_frame_data", False)
        self.repeat: bool = kwargs.get("repeat", False)
        self.antialias: bool = kwargs.get("antialias", True)
        self.instrument_view: bool = kwargs.get("instrument_view", False)
        self.draw_cache_mode: bool = kwargs.get("draw_cache_mode", False)

        # debug
        self.quiet = kwargs.get("quiet", False)
        self.instrument_view &= not self.quiet

        # color cycle list
        self.color_cycle = self.__color_cycle()


    def show(self) -> None:
        self.app.exec_()  # or app.exec_() for PyQt5 / PySide2


    def __color_cycle(self) -> Generator[str, None, None]:
        """This method is a Generator which handles the color cycle of line / scatter
        plots which do not specify a color."""

        while True:
            for _, v in mplcolors.TABLEAU_COLORS.items():
                yield str(v)


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
                anchor=(0, 3 - spacing*ntext),
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

        vector_item: GraphItem = pg.GraphItem()
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
        view.addItem(vector_item) #type:ignore

    # --------------------------------------------------------------------------------------

    def setup(self) -> None:
        self.istep = 0

        ## Always start by initializing Qt (only once per application)
        if self.quiet:
            self.app = QtCore.QCoreApplication([])  # no GUI
            return
        else:
            self.app = QtWidgets.QApplication([])   # GUI

        # enable anti-aliasing
        pg.setConfigOptions(antialias=self.antialias)

        self.wins: list[GraphicsLayoutWidget] = []     # contains view layouts for each simobj
        self.shortcuts = []


    def add_simobject(self, simobj: SimObject) -> None:

            if self.quiet:
                return

            self.simobjs += [simobj]

            # setup window for each simobj
            _win = GraphicsLayoutWidget()
            _win.resize(*(np.array([*self.figsize]) * 100))
            _win.show()
            self.wins += [_win]

            # shortcut keys callbacks for each simobj view
            _shortcut = QtWidgets.QShortcut(QKeySequence("Q"), _win)
            _shortcut.activated.connect(self.close_windows) #type:ignore
            self.shortcuts += [_shortcut]

            # setup user-defined plots for each simobj
            for i, (title, axes) in enumerate(simobj.plot.get_config().items()):
                _plot_item = _win.addPlot(row=i, col=0, colspan=2, title=title, name=title)   # add PlotItem to View
                _plot_item.showGrid(True, True, 0.5)    # enable grid
                _aspect = axes.get("aspect", self.aspect)   # look for aspect in plot config; default to class init
                _plot_item.setAspectLocked(_aspect == "equal")
                _pen = {"color": simobj.plot.color_code, "width": simobj.size}
                _graphic_item = PlotDataItem(x=[], y=[], pen=_pen,
                                             useCache=self.draw_cache_mode,
                                             antialias=self.antialias,
                                             autoDownsample=True,
                                             downsampleMethod="peak",
                                             clipToView=True,
                                             skipFiniteCheck=True,
                                             )
                _plot_item.addItem(_graphic_item)   # init PlotCurve
                simobj.plot.qt_traces += [_graphic_item]   # add GraphicsItem reference to SimObject

            # setup vehicle viewer widget
            if self.instrument_view:
                _view = ViewBox(name="instrument_view")
                _view.setAspectLocked(True)
                _view.setRange(xRange=[-1,1], yRange=[-1, 1])
                _win.addItem(_view, row=(i + 1), col=0, colspan=1) #type:ignore

                # ViewBox for text
                _text_view = ViewBox()
                _text_view.setRange(xRange=[-1, 1], yRange=[-1, 1])
                _win.addItem(_text_view, row=(i + 1), col=1, colspan=1) #type:ignore

                self.attitude_graph_item: GraphItem = pg.GraphItem()
                _view.addItem(self.attitude_graph_item) #type:ignore

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


    def FuncAnimation(self,
                      func: Callable,
                      frames: Callable|Generator|int,
                      interval: int,
                      ) -> None:
        # plotter.widget
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(partial(func, frame=0))
        self.timer.start(interval)


    # TODO this may belong in Sim class...
    def _animate_func(self, frame, simobj: SimObject, step_func: Callable, moving_bounds: bool = False):

        # # TEMP #############################################
        # # %-error time profile of pyqtgraph painting process
        # if self.instrument_view and (self.istep % 10) == 0:
        #     perr = abs((time.time() - self._tstart) - self.dt) / self.dt
        #     if (ti := self.get_text_item(0, 0)):
        #         ti.setText(f"{np.round(perr, 2)}")
        # self._tstart = time.time()

        self.istep += 1

        # run ODE solver step
        step_func(istep=self.istep)

        # handle plot axes boundaries
        # self.update_axes_boundary(
        #         self.ax,
        #         pos=(xdata[-1], ydata[-1]),
        #         moving_bounds=moving_bounds
        #         )

        for subplot_id in range(len(simobj.plot.get_config())):
            # get data from SimObject based on state_select user configuration
            xdata, ydata = simobj.get_plot_data(subplot_id, self.istep)
            # pen = simobj.plot._get_qt_pen(subplot_id=subplot_id)
            pen = {"color": simobj.plot.color_code, "width": simobj.plot.size}
            simobj._update_patch_data(xdata, ydata, pen=pen, subplot_id=subplot_id)

        # drawing the instrument view of vehicle
        # TODO generalize: each simobj has its own body to draw.
        if self.instrument_view:
            self.__draw_instrument_view(simobj)

        # TODO run post-animation func when finished
        if self.istep >= self.Nt:
            self.exit()


    def close_windows(self) -> None:
        for win in self.wins:
            win.close()


    def exit(self) -> None:
        if self.quiet:
            self.app.exit()
        else:
            # stop timer and close all open windows
            self.timer.stop()


    def __draw_instrument_view(self, _simobj: SimObject) -> None:
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
        quat_ids = _simobj.model.get_state_id(["q0", "q1", "q2", "q3"])
        istate = _simobj.model.get_current_state()
        iquat = istate[quat_ids]

        # get rotation matrix
        _iquat = quaternion.from_float_array(iquat)
        dcm = quaternion.as_rotation_matrix(_iquat).flatten()

        # swap rows 1 & 2; swap cols 1 & 2
        yz_swapped_dcm = np.array([dcm[0], dcm[2], dcm[1],
                                dcm[6], dcm[8], dcm[7],
                                dcm[3], dcm[5], dcm[4]])

        transform = QTransform(*yz_swapped_dcm)
        self.attitude_graph_item.setTransform(transform)


    def _time_slider_update(self, val: float, _simobjs: list[SimObject]) -> None:
        pass


    # def set_lim(self, lim: list|tuple, padding=0.02) -> None:
    #     assert len(lim) == 4

    #     x = lim[0]
    #     y = lim[2]
    #     width = lim[1] - lim[0]
    #     height = lim[3] - lim[2]

    #     newRect = QRectF(x, y, width, height) #type:ignore
    #     self.widget.setRange(newRect, padding=padding)


    # def __x_axis_right_border_append(self, ax: Axes, val: float):
    #     pass


    # def __x_axis_left_border_append(self, ax: Axes, val: float):
    #     pass


    # def __y_axis_top_border_append(self, ax: Axes, val: float):
    #     pass


    # def __y_axis_bottom_border_append(self, ax: Axes, val: float):
    #     pass


    # def update_axes_boundary(self, ax: Axes, pos: list|tuple, margin: float = 0.1, moving_bounds: bool = False) -> None:
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


    # def plot(self,
    #          x: np.ndarray|list,
    #          y: np.ndarray|list,
    #          color: str = "",
    #          linestyle: str = "",
    #          linewidth: float = 3,
    #          marker: Optional[str] = None,
    #          **kwargs):

    #     # convert mpl color to rgb
    #     if color:
    #         color_code = mplcolors.TABLEAU_COLORS[color]
    #     else:
    #         color_code = next(self.color_cycle)

    #     # convert mpl color to rgb
    #     rgb_color = mplcolors.to_rgb(color_code)
    #     rgb_color = (rgb_color[0]*255, rgb_color[1]*255, rgb_color[2]*255)

    #     line = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(rgb_color, width=linewidth), symbol=marker)
    #     self.widget.addItem(line)


    # def scatter(self,
    #             x: np.ndarray|list,
    #             y: np.ndarray|list,
    #             color: str = "",
    #             linewidth: float = 1,
    #             marker: str = "o",
    #             **kwargs):

    #     # convert mpl color to rgb
    #     if color:
    #         color_code = mplcolors.TABLEAU_COLORS[color]
    #     else:
    #         color_code = next(self.color_cycle)

    #     # convert mpl color to rgb
    #     rgb_color = mplcolors.to_rgb(color_code)
    #     rgb_color = (rgb_color[0]*255, rgb_color[1]*255, rgb_color[2]*255)

    #     scatter = pg.ScatterPlotItem(x=x, y=y, pen=pg.mkPen(rgb_color, width=linewidth), symbol=marker)
    #     self.widget.addItem(scatter)


    # def autoscale(self, xdata: np.ndarray|list, ydata: np.ndarray|list) -> None:
    #     # autoscale
    #     self.set_lim([min(xdata) - 0.2, max(xdata) + 0.2, min(ydata) - 0.2, max(ydata) + 0.2])

