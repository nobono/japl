import numpy as np

from typing import Callable
from typing import Generator

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation as MplFuncAnimation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from japl.SimObject.SimObject import SimObject

# ---------------------------------------------------



class Plotter:

    def __init__(self, Nt: int, **kwargs) -> None:

        self.Nt = Nt
        self.simobjs = []

        # plotting
        self.figsize = kwargs.get("figsize", (6, 4))
        self.aspect: float|str = kwargs.get("aspect", "equal")
        self.blit: bool = kwargs.get("blit", False)
        self.cache_frame_data: bool = kwargs.get("cache_frame_data", False)
        self.repeat: bool = kwargs.get("repeat", False)
        self.antialias: bool = kwargs.get("antialias", True)
        self.instrument_view: bool = kwargs.get("instrument_view", False)


    def add_text(self, text: str, window_id: int = 0, color: tuple = (255, 255, 255),
                 spacing: float = 0.6) -> None:
         pass


    def setup(self) -> None:

        # instantiate figure and axes
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # set aspect initial ratio
        self.ax.set_aspect(self.aspect)


    def add_simobject(self, simobj: SimObject) -> None:

        self.simobjs += [simobj]

        # add simobj patch to Sim axes
        for simobj in self.simobjs:
            for i, (title, axes) in enumerate(simobj.plot.get_config().items()):
                _width = simobj.plot.size
                _color = simobj.plot.color
                _graphic_item = Line2D([], [], color=_color, linewidth=_width, antialiased=self.antialias)
                simobj.plot.traces += [_graphic_item]
                self.ax.add_line(_graphic_item)


    def show(self, block: bool = True) -> None:
        return plt.show(block=block)


    def autoscale(self, xdata: np.ndarray|list, ydata: np.ndarray|list) -> None:
        # autoscale
        self.ax.set_xlim([min(xdata) - 0.2, max(xdata) + 0.2])
        self.ax.set_ylim([min(ydata) - 0.2, max(ydata) + 0.2])


    def FuncAnimation(self,
                      func: Callable,
                      frames: Callable|Generator|int,
                      interval: int|float,
                      ):

        anim = MplFuncAnimation(
                fig=self.fig,
                func=func,
                frames=frames,
                interval=interval,
                blit=self.blit,
                cache_frame_data=self.cache_frame_data,
                repeat=self.repeat,
                )

        return anim


    def _animate_func(self, frame, simobj: SimObject, step_func: Callable, moving_bounds: bool = False):

        self.istep = frame + 1

        # run post-animation func when finished
        if self.istep >= self.Nt:
            self._post_anim_func(self.simobjs)

        # run ODE solver step
        step_func(istep=self.istep)

        # update SimObject data
        for subplot_id in range(len(simobj.plot.get_config())):
            # get data from SimObject based on state_select user configuration
            xdata, ydata = simobj.get_plot_data(subplot_id, self.istep)
            simobj._update_patch_data(xdata, ydata, subplot_id=subplot_id)

            # # exit on exception
            # if len(xdata) == 0:
            #     return []

            # handle plot axes boundaries
            self.update_axes_boundary(
                    self.ax,
                    pos=(xdata[-1], ydata[-1]),
                    moving_bounds=moving_bounds
                    )

        # TODO this needs to account for several axes to plot on...
        return simobj.plot.traces


    def _post_anim_func(self, _simobjs: list[SimObject]) -> None:
        """
            This method is the post-animation function which runs at the end of the
        FuncAnimation method. This method sets up the time-slider on the plot axes and configures
        the time-slider callback function.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- _simobjs - list of SimObject
        -------------------------------------------------------------------
        """

        if "time_slider" not in dir(self):
            self.setup_time_slider(self.Nt, _simobjs=_simobjs)


    def _time_slider_update(self, val: float, _simobjs: list[SimObject]) -> None:

        for _simobj in _simobjs:
            # get data range
            val = int(val)

            # # exit on exception
            # if len(xdata) == 0:
            #     return

            # update artist data
            for subplot_id in range(len(_simobj.plot.get_config())):
                # select user specficied state(s)
                xdata, ydata = _simobj.get_plot_data(subplot_id, val)
                _simobj._update_patch_data(xdata, ydata, subplot_id=subplot_id)



    def __x_axis_right_border_append(self, ax: Axes, val: float):
        _xlim = ax.get_xlim()
        ax.set_xlim((_xlim[0], _xlim[1] + val))


    def __x_axis_left_border_append(self, ax: Axes, val: float):
        _xlim = ax.get_xlim()
        ax.set_xlim((_xlim[0] + val, _xlim[1]))


    def __y_axis_top_border_append(self, ax: Axes, val: float):
        _ylim = ax.get_ylim()
        ax.set_ylim((_ylim[0], _ylim[1] + val))


    def __y_axis_bottom_border_append(self, ax: Axes, val: float):
        _ylim = ax.get_ylim()
        ax.set_ylim((_ylim[0] + val, _ylim[1]))


    def update_axes_boundary(self, ax: Axes, pos: list|tuple, margin: float = 0.1, moving_bounds: bool = False) -> None:
        """
            This method handles the plot axes boundaries during FuncAnimation frames.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- ax - matplotlib Axes object
        -- pos - xy position of Artist being plotted
        -- margin - % margin value between Artist xy position and Axes border
        -------------------------------------------------------------------
        """

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xcenter, ycenter = pos

        xlen = xlim[1] - xlim[0]
        ylen = ylim[1] - ylim[0]

        aspect_ratio = 1.4
        X_RANGE_LIM = 100
        Y_RANGE_LIM = int(X_RANGE_LIM / aspect_ratio)

        # weight the amount to move the border proportional to how close
        # the object cetner is to the border limit. Also account for the 
        # scale of the current plot window (xlen, ylen) in the amount to
        # change the current boundary.

        if (weight := xcenter - xlim[1]) + margin > 0:
            length_weight = min(xlen, 20)
            self.__x_axis_right_border_append(ax, margin * length_weight * abs(weight))
            if xlen > X_RANGE_LIM and moving_bounds:
                self.__x_axis_left_border_append(ax, margin * length_weight * abs(weight))

        if (weight := xcenter - xlim[0]) - margin < 0:
            length_weight = min(xlen, 20)
            self.__x_axis_left_border_append(ax, -(margin * length_weight * abs(weight)))
            if xlen > X_RANGE_LIM and moving_bounds:
                self.__x_axis_right_border_append(ax, -(margin * length_weight * abs(weight)))

        if (weight := ycenter - ylim[1]) + margin > 0:
            length_weight = min(ylen, 20)
            self.__y_axis_top_border_append(ax, margin * length_weight * abs(weight))
            if ylen > Y_RANGE_LIM and moving_bounds:
                self.__y_axis_bottom_border_append(ax, margin * length_weight * abs(weight))

        if (weight := ycenter - ylim[0]) - margin < 0:
            length_weight = min(ylen, 20)
            self.__y_axis_bottom_border_append(ax, -(margin * length_weight * abs(weight)))
            if ylen > Y_RANGE_LIM and moving_bounds:
                self.__y_axis_top_border_append(ax, -(margin * length_weight * abs(weight)))


    def setup_time_slider(self, Nt: int, _simobjs: list[SimObject]) -> None:

        axis_position = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='white') # type:ignore
        self.time_slider = Slider(
            axis_position,
            label='Time (s)',
            valmin=0,
            valmax=Nt,
            valinit=0
            )
        self.time_slider.on_changed(lambda t: self._time_slider_update(t, _simobjs=_simobjs))
        self.time_slider.set_val(Nt) # initialize slider at end-time


    def exit(self) -> None:
        pass

