# ---------------------------------------------------
from typing import Callable
from typing import Generator
from typing import Optional

from tqdm import tqdm

import numpy as np

from japl.SimObject.SimObject import SimObject

from scipy.integrate import solve_ivp

from functools import partial

from scipy import constants

# ---------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from japl.DeviceInput.DeviceInput import DeviceInput



class Sim:

    def __init__(self,
                 t_span: list|tuple,
                 dt: float,
                 simobjs: list[SimObject],
                 events: list = [],
                 animate: bool|int = False,
                 **kwargs,
                 ) -> None:

        self.t_span = t_span
        self.dt = dt
        self.simobjs = simobjs
        self.events = events
        self.animate = bool(animate) # choice of iterating solver over each dt step

        # setup time array
        self.istep = 0
        self.Nt = int(self.t_span[1] / self.dt)
        self.t_array = np.linspace(self.t_span[0], self.t_span[1], self.Nt + 1)
        self.T = np.array([])

        # ODE solver params
        self.rtol: float = kwargs.get("rtol", 1e-6)
        self.atol: float = kwargs.get("atol", 1e-6)
        self.max_step: float = kwargs.get("max_step", 0.2)

        # plotting
        self.aspect: float|str = kwargs.get("aspect", "equal")
        self.blit: bool = kwargs.get("blit", False)
        self.cache_frame_data: bool = kwargs.get("cache_frame_data", False)
        self.repeat: bool = kwargs.get("repeat", False)
        self.autoscale: bool = kwargs.get("plot_follow", False)

        # device inputs
        self.use_device_input = kwargs.get("use_device_input", False)
        self.device_input = DeviceInput()


    def run(self) -> "Sim":

        simobj = self.simobjs[0]

        # begin device input read thread
        if self.use_device_input:
            self.device_input.start()

        # instantiate figure and axes
        self.fig, self.ax = plt.subplots(figsize=(6, 4))

        # set aspect initial ratio
        self.ax.set_aspect(self.aspect)

        # add simobj patch to Sim axes
        self.ax.add_patch(simobj.plot.patch)
        self.ax.add_line(simobj.plot.trace)


        if not self.animate:
            ################################
            # solver
            ################################

            sol = solve_ivp(
                    fun=self.step,
                    t_span=self.t_span,
                    t_eval=self.t_array,
                    y0=simobj.X0,
                    args=(simobj,),
                    events=self.events,
                    rtol=self.rtol,
                    atol=self.atol,
                    max_step=self.max_step,
                    )
            self.T = sol['t']
            simobj.Y = sol['y'].T
            simobj._set_T_array_ref(self.T) # simobj.T reference to sim.T

            xdata, ydata = simobj.get_plot_data()
            simobj._update_patch_data(xdata, ydata)

            # autoscale
            self.ax.set_xlim([min(xdata) - 0.2, max(xdata) + 0.2])
            self.ax.set_ylim([min(ydata) - 0.2, max(ydata) + 0.2])

            self._setup_time_slider(self.Nt, [simobj])

        elif self.animate:

            ################################
            # solver for one step at a time
            ################################

            # pre-allocate output arrays
            self.T = np.zeros((self.Nt, ))
            simobj.Y = np.zeros((self.Nt, len(simobj.X0)))
            simobj.Y[0] = simobj.X0
            simobj._set_T_array_ref(self.T) # simobj.T reference to sim.T

            # try to set animation frame intervals to real time
            interval = int(max(1, self.dt * 1000))

            anim = FuncAnimation(
                    fig=self.fig,
                    func=partial(self._animate_func, _simobj=simobj),
                    frames=partial(self._frames, _simobj=simobj),
                    interval=interval,
                    blit=self.blit,
                    cache_frame_data=self.cache_frame_data,
                    repeat=self.repeat,
                    )

        plt.show()

        return self


    def step(self, t, X, simobj):
        # ac = np.array([3, .5*np.sin(1 * t), 0])

        ac = np.array([0, 0, 0])

        # get device input
        if self.use_device_input:
            (lx, ly, rx, ry) = self.device_input.get()
            ac = ac + np.array([100*lx, 0, 100*ly])

        fuel_burn = X[6]
        # if fuel_burn < 100:
        #     ac[0] += 20
        #     ac[2] += 40

        burn_const = 0.4

        U = np.array([*ac])
        Xdot = simobj.step(X, U)
        Xdot[6] = burn_const * np.linalg.norm(ac)

        return Xdot


    def _animate_func(self, frame, _simobj: SimObject):
        xdata, ydata = frame

        # exit on exception
        if len(xdata) == 0:
            return []

        _simobj._update_patch_data(xdata, ydata)

        # handle plot axes boundaries
        self.__update_axes_boundary(self.ax, pos=(xdata[-1], ydata[-1]), autoscale=self.autoscale)

        return [_simobj.plot.patch, _simobj.plot.trace]


    def _frames(self, _simobj: SimObject):
        """
            This method is a Generator function which passes frame data to
        FuncAnimation. Take SimObject and returns iterable of matplotlib artist

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- _simobjs - list of SimObject
        -------------------------------------------------------------------
        """

        while self.istep < self.Nt - 1:

            self.istep += 1
            self._step_solve_ivp(self.istep, _simobj, rtol=self.rtol, atol=self.atol, max_step=self.max_step)

            # get data from SimObject based on state_select user configuration
            xdata, ydata = _simobj.get_plot_data(self.istep)
            yield (xdata, ydata)

        self._post_anim_func(self.simobjs)


    def _step_solve_ivp(self,
                        istep: int,
                        _simobj: SimObject,
                        rtol: float = 1e-6,
                        atol: float = 1e-6,
                        max_step: float = 0.2
                        ) -> None:
        """
            This method is an update step for the ODE solver from time step 't' to 't + dt';
        used by FuncAnimation.

        -------------------------------------------------------------------
        -- Arguments
        -------------------------------------------------------------------
        -- istep - integer step
        -- _simobj - SimObject
        -- rtol - relative tolerance for ODE Solver
        -- atol - absolute tolerance for ODE Solver
        -- max_step - max step size for ODE Solver
        -------------------------------------------------------------------
        -------------------------------------------------------------------
        -- Returns:
        -------------------------------------------------------------------
        --- t - time array of solution points
        --- y - (Nt x N_state) array of solution points from ODE solver
        -------------------------------------------------------------------

        """
        tstep_prev = self.t_array[istep - 1]
        tstep = self.t_array[istep]
        x0 = _simobj.Y[istep - 1]

        sol = solve_ivp(
                fun=self.step,
                t_span=(tstep_prev, tstep),
                t_eval=[tstep],
                y0=x0,
                args=(_simobj,),
                events=self.events,
                rtol=rtol,
                atol=atol,
                max_step=max_step
                )
        self.T[istep] = sol['t'][0]
        _simobj.Y[istep] = sol['y'].T[0]


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
            self._setup_time_slider(self.Nt, _simobjs=_simobjs)

    def _time_slider_update(self, val: float, _simobjs: list[SimObject]) -> None:

        for _simobj in _simobjs:
            # get data range
            val = int(val)

            # select user specficied state(s)
            xdata, ydata = _simobj.get_plot_data(val)

            # exit on exception
            if len(xdata) == 0:
                return

            # update artist data
            _simobj._update_patch_data(xdata, ydata)



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


    def __update_axes_boundary(self, ax: Axes, pos: list|tuple, margin: float = 0.1, autoscale: bool = False) -> None:
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
            if xlen > X_RANGE_LIM and not autoscale:
                self.__x_axis_left_border_append(ax, margin * length_weight * abs(weight))

        if (weight := xcenter - xlim[0]) - margin < 0:
            length_weight = min(xlen, 20)
            self.__x_axis_left_border_append(ax, -(margin * length_weight * abs(weight)))
            if xlen > X_RANGE_LIM and not autoscale:
                self.__x_axis_right_border_append(ax, -(margin * length_weight * abs(weight)))

        if (weight := ycenter - ylim[1]) + margin > 0:
            length_weight = min(ylen, 20)
            self.__y_axis_top_border_append(ax, margin * length_weight * abs(weight))
            if ylen > Y_RANGE_LIM and not autoscale:
                self.__y_axis_bottom_border_append(ax, margin * length_weight * abs(weight))

        if (weight := ycenter - ylim[0]) - margin < 0:
            length_weight = min(ylen, 20)
            self.__y_axis_bottom_border_append(ax, -(margin * length_weight * abs(weight)))
            if ylen > Y_RANGE_LIM and not autoscale:
                self.__y_axis_top_border_append(ax, -(margin * length_weight * abs(weight)))


    def _setup_time_slider(self, Nt: int, _simobjs: list[SimObject]) -> None:

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


