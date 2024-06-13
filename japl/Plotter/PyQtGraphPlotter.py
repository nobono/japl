from functools import partial
import numpy as np

from typing import Callable, Optional
from typing import Generator


from japl.SimObject.SimObject import SimObject

import pyqtgraph as pg
from pyqtgraph import PlotCurveItem, QtGui, mkPen
from pyqtgraph import QtWidgets
from pyqtgraph import PlotWidget
from pyqtgraph.Qt import QtCore
from pyqtgraph.Qt.QtCore import QRectF
from pyqtgraph.Qt.QtGui import QKeySequence

from matplotlib import colors as mplcolors
# ---------------------------------------------------



class PyQtGraphPlotter:

    def __init__(self, Nt: int, figsize: tuple = (6, 4), **kwargs) -> None:

        self.Nt = Nt
        self.figsize = figsize
        self.simobjs = []

        # plotting
        self.aspect: float|str = kwargs.get("aspect", "equal")
        self.blit: bool = kwargs.get("blit", False)
        self.cache_frame_data: bool = kwargs.get("cache_frame_data", False)
        self.repeat: bool = kwargs.get("repeat", False)
        self.antialias: bool = kwargs.get("antialias", True)

        # color cycle list
        self.color_cycle = self.__color_cycle()


    def __color_cycle(self) -> Generator[str, None, None]:
        """This method is a Generator which handles the color cycle of line / scatter
        plots which do not specify a color."""

        while True:
            for _, v in mplcolors.TABLEAU_COLORS.items():
                yield str(v)


    def setup(self, simobjs: list[SimObject]):
        self.simobjs = simobjs
        self.istep = 0

        ## Always start by initializing Qt (only once per application)
        self.app = QtWidgets.QApplication([])
        self.win = QtWidgets.QMainWindow()
        self.widget = PlotWidget()

        # enable anti-aliasing
        pg.setConfigOptions(antialias=self.antialias)

        # set apsect
        self.widget.setAspectLocked(self.aspect == "equal")

        # enable grid
        self.widget.showGrid(True, True, 0.5)

        # setup window
        self.win.setCentralWidget(self.widget)
        self.win.resize(*(np.array([*self.figsize]) * 100))
        self.win.show()

        # shortcut keys callbacks
        self.shortcut = QtWidgets.QShortcut(QKeySequence("Q"), self.win)
        self.shortcut.activated.connect(self.win.close) #type:ignore

        for simobj in self.simobjs:
            simobj.plot.add_patch_to_plot(self.widget)


    def show(self) -> None:
        self.app.exec()  # or app.exec_() for PyQt5 / PySide2


    def plot(self,
             x: np.ndarray|list,
             y: np.ndarray|list,
             color: str = "",
             linestyle: str = "",
             linewidth: float = 3,
             marker: Optional[str] = None,
             **kwargs):

        # convert mpl color to rgb
        if color:
            color_code = mplcolors.TABLEAU_COLORS[color]
        else:
            color_code = next(self.color_cycle)

        # convert mpl color to rgb
        rgb_color = mplcolors.to_rgb(color_code)
        rgb_color = (rgb_color[0]*255, rgb_color[1]*255, rgb_color[2]*255)

        line = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(rgb_color, width=linewidth), symbol=marker)
        self.widget.addItem(line)


    def scatter(self,
                x: np.ndarray|list,
                y: np.ndarray|list,
                color: str = "",
                linewidth: float = 1,
                marker: str = "o",
                **kwargs):

        # convert mpl color to rgb
        if color:
            color_code = mplcolors.TABLEAU_COLORS[color]
        else:
            color_code = next(self.color_cycle)

        # convert mpl color to rgb
        rgb_color = mplcolors.to_rgb(color_code)
        rgb_color = (rgb_color[0]*255, rgb_color[1]*255, rgb_color[2]*255)

        scatter = pg.ScatterPlotItem(x=x, y=y, pen=pg.mkPen(rgb_color, width=linewidth), symbol=marker)
        self.widget.addItem(scatter)


    def autoscale(self, xdata: np.ndarray|list, ydata: np.ndarray|list) -> None:
        # autoscale
        self.set_lim([min(xdata) - 0.2, max(xdata) + 0.2, min(ydata) - 0.2, max(ydata) + 0.2])


    def FuncAnimation(self,
                      func: Callable,
                      frames: Callable|Generator|int,
                      interval: int,
                      ) -> None:
        # plotter.widget
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(partial(func, frame=0))
        self.timer.start(interval)


    def _animate_func(self, frame, _simobj: SimObject, step_func: Callable, moving_bounds: bool = False):

        self.istep += 1

        # run post-animation func when finished
        if self.istep >= self.Nt:
            # self._post_anim_func(self.simobjs)
            self.timer.stop()

        # run ODE solver step
        step_func(istep=self.istep)

        # get data from SimObject based on state_select user configuration
        xdata, ydata = _simobj.get_plot_data(self.istep)

        # exit on exception
        if len(xdata) == 0:
            return []

        # handle plot axes boundaries
        # self.update_axes_boundary(
        #         self.ax,
        #         pos=(xdata[-1], ydata[-1]),
        #         moving_bounds=moving_bounds
        #         )

        pen = _simobj.plot._get_qt_pen()
        _simobj._update_patch_data(xdata, ydata, pen=pen)


    def _time_slider_update(self, val: float, _simobjs: list[SimObject]) -> None:
        pass


    def set_lim(self, lim: list|tuple, padding=0.02) -> None:
        assert len(lim) == 4

        x = lim[0]
        y = lim[2]
        width = lim[1] - lim[0]
        height = lim[3] - lim[2]

        newRect = QRectF(x, y, width, height) #type:ignore
        self.widget.setRange(newRect, padding=padding)


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


