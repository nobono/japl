# ---------------------------------------------------
from typing import Callable
from typing import Generator
from typing import Optional

from tqdm import tqdm

import numpy as np

from japl.SimObject.SimObject import SimObject

from scipy.integrate import solve_ivp

from functools import partial

# ---------------------------------------------------

from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D



class Sim:

    def __init__(self,
                 t_span: list|tuple,
                 dt: float,
                 simobjs: list[SimObject],
                 events: list = [],
                 anim_solve: bool = False,
                 **kwargs,
                 ) -> None:
        self.t_span = t_span
        self.dt = dt
        self.simobjs = simobjs
        self.events = events
        self.anim_solve = anim_solve # choice of iterating solver over each dt step

        # setup time array
        self.istep = 0
        self.Nt = int(self.t_span[1] / self.dt)
        self.t_array = np.linspace(self.t_span[0], self.t_span[1], self.Nt + 1)
        self.T = np.array([])

        # plotting
        self.aspect: float|str = kwargs.get("aspect", "equal")


    def step(self, t, X, simobj):
        ac = np.array([3*np.cos(2 * t), .5*np.sin(1 * t), 0])

        fuel_burn = X[6]
        if fuel_burn >= 100:
            ac = np.zeros((3,))

        burn_const = 0.4

        U = np.array([*ac])
        Xdot = simobj.step(X, U)
        Xdot[6] = burn_const * np.linalg.norm(ac)

        return Xdot


    def __call__(self) -> "Sim":

        simobj = self.simobjs[0]

        ################################
        # solver
        ################################

        if not self.anim_solve:
            sol = solve_ivp(
                    fun=self.step,
                    t_span=self.t_span,
                    t_eval=self.t_array,
                    y0=simobj.X0,
                    args=(simobj,),
                    events=self.events,
                    rtol=1e-3,
                    atol=1e-6,
                    max_step=0.2,
                    )
            self.T = sol['t']
            simobj.Y = sol['y'].T

        ################################
        # solver for one step at a time
        ################################

        elif self.anim_solve:

            # pre-allocate output arrays
            self.T = np.zeros((self.Nt, ))
            simobj.Y = np.zeros((self.Nt, len(simobj.X0)))
            simobj._set_T_array_ref(self.T) # simobj.T reference to sim.T

            self.fig, self.ax = plt.subplots(figsize=(6, 4))

            self.ax.set_aspect(self.aspect)

            # add simobj patch to Sim axes
            self.ax.add_patch(simobj.plot.patch)
            self.ax.add_line(simobj.plot.trace)

            anim = FuncAnimation(
                    fig=self.fig,
                    func=partial(self.animate, _simobj=simobj),
                    frames=partial(self._frames, _simobj=simobj),
                    interval=int(max(1, self.dt * 1000)),
                    blit=False,
                    cache_frame_data=False
                    )

            plt.show()

        return self


    def animate(self, frame, _simobj: SimObject):
        xdata, ydata = frame

        # if (state_select := _simobj.plot.state_select):
        # plot trace data
        # xdata = y[:, 0]
        # ydata = y[:, 1]

        _simobj.plot.trace.set_data(xdata, ydata)

        # plot current step position data
        xcenter = xdata[-1]
        ycenter = ydata[-1]

        _simobj.plot.patch.set_center((xcenter, ycenter))

        # handle plot axes boundaries
        xmargin = .2
        ymargin = .2
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        if xcenter - xlim[1] + xmargin > 0:
            self.ax.set_xlim([xlim[0], xcenter + xmargin])
        if xcenter - xlim[0] - xmargin < 0:
            self.ax.set_xlim(([xcenter - xmargin, xlim[1]]))

        if ycenter - ylim[1] + ymargin > 0:
            self.ax.set_ylim([ylim[0], ycenter + ymargin])
        if ycenter - ylim[0] - ymargin < 0:
            self.ax.set_ylim([ycenter - ymargin, ylim[1]])

        # xlim = self.ax.get_xlim()
        # ylim = self.ax.get_ylim()
        # xrange = np.abs(xlim[1] - xlim[0])
        # yrange = ylim[1] - ylim[0]

        # RANGE_LOCK = 3
        # if xrange > RANGE_LOCK:
        #     diff = (np.abs(xdata - xlim[0]) + np.abs(xdata - xlim[1]) - RANGE_LOCK) / 2
        #     self.ax.set_xlim([xlim[0] + diff, xlim[1] + diff])
        # if yrange > RANGE_LOCK:
        #     diff = (np.abs(ydata - ylim[0]) + np.abs(ydata - ylim[1]) - RANGE_LOCK) / 2
        #     self.ax.set_ylim([ylim[0] + diff, ylim[1] - diff])
        # if yrange < -RANGE_LOCK:
        #     diff = (np.abs(ydata - ylim[0]) + np.abs(ydata - ylim[1]) + RANGE_LOCK) / 2
        #     self.ax.set_ylim([ylim[0] - diff, ylim[1] - diff])

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
            self._anim_update(self.istep, _simobj)

            if (state_select := _simobj.plot.state_select):
                xdata = _simobj.get_data(self.istep, state_select["x"])
                ydata = _simobj.get_data(self.istep, state_select["y"])
                yield (xdata, ydata)

        self._post_anim_func(self.simobjs)


    def _anim_update(self, istep: int, _simobj: SimObject, rtol: float = 1e-6, atol: float = 1e-6) -> None:
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
            axis_position = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='white') # type:ignore
            self.time_slider = Slider(
                axis_position,
                label='Time (s)',
                valmin=0,
                valmax=self.Nt,
                valinit=0
                )
            self.time_slider.on_changed(lambda t: self._time_slider_update(t, _simobjs=_simobjs))
            self.time_slider.set_val(self.Nt) # initialize slider at end-time

        # self.ax.margins(.05, .05)
        # self.ax.autoscale()


    def _time_slider_update(self, val: float, _simobjs: list[SimObject]) -> None:

        for _simobj in _simobjs:
            # get data range
            val = int(val)
            # t = self.T[:val]
            # y = _simobj.Y[:val]

            # select user specficied state(s)
            if (state_select := _simobj.plot.state_select):
                # xdata = t
                # ydata = y[:, 1]
                xdata = _simobj.get_data(val, state_select["x"])
                ydata = _simobj.get_data(val, state_select["y"])

                # update artist data
                _simobj.plot.patch.set_center((xdata[-1], ydata[-1]))
                _simobj.plot.trace.set_data(xdata, ydata)


