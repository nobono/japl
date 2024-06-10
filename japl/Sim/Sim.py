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



class Sim:

    def __init__(self,
                 t_span: list|tuple,
                 dt: float,
                 simobjs: list[SimObject],
                 events: list = [],
                 anim_solve: bool = False,
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

    # def _setup(self):

    #     simobj = self.simobjs[0]
    #     # x0 = simobj.X0

    #     # setup time array
    #     Nt = int(self.t_span[1] / self.dt)
    #     t_array = np.linspace(self.t_span[0], self.t_span[1], Nt)

    #     # pre-allocate output arrays
    #     # simobj.T = np.zeros(t_array.shape)
    #     simobj.T = t_array
    #     simobj.Y = np.zeros((t_array.shape[0], simobj.X0.shape[0]))

    #     # initial state outputs
    #     simobj.T[0] = self.t_span[0]
    #     simobj.Y[0] = simobj.X0


    def step(self, t, X, simobj):
        ac = np.array([1, 5*np.sin(.1*t), 0])

        fuel_burn = X[6]
        if fuel_burn >= 100:
            ac = np.zeros((3,))

        burn_const = 0.4

        U = np.array([*ac])
        Xdot = simobj.step(X, U)
        Xdot[6] = burn_const * np.linalg.norm(ac)

        return Xdot



    def __call__(self):

        simobj = self.simobjs[0]
        # x0 = simobj.X0

        # pre-allocate output arrays
        # simobj.T = np.zeros(t_array.shape)
        # simobj.T = t_array
        # simobj.Y = np.zeros((t_array.shape[0], simobj.X0.shape[0]))

        # initial state outputs
        # simobj.T[0] = self.t_span[0]
        # simobj.Y[0] = simobj.X0

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
            simobj.T = sol['t']
            simobj.Y = sol['y'].T

        ################################
        # solver for one step at a time
        ################################

        elif self.anim_solve:

            simobj.T = np.zeros((self.Nt, ))
            simobj.Y = np.zeros((self.Nt, len(simobj.X0)))

            ####################################
            # TEMP
            ####################################

            # plt.show(block=False)
            # self.ax.add_patch(simobj.patch)

            # FuncAnimation
            # def anim_update(args):
            #     return plt.scatter(0, args)

            # def frames():
            #     while True:
            #         yield regr_magic()

            ####################################

            # for istep, (tstep_prev, tstep) in tqdm(enumerate(zip(t_array, t_array[1:])), total=Nt):

            #     sol = solve_ivp(
            #             fun=self.step,
            #             t_span=(tstep_prev, tstep),
            #             t_eval=[tstep],
            #             y0=x0,
            #             args=(simobj,),
            #             events=self.events,
            #             rtol=1e-3,
            #             atol=1e-6,
            #             )

            #     # check for stop event
            #     # if check_for_events(sol['t_events']):
            #     #     # truncate output arrays if early stoppage
            #     #     T = T[:istep + 1]
                #     Y = Y[:istep + 1]
            #     #     break
            #     # else:
            #     # store output
            #     t = sol['t'][0]
            #     y = sol['y'].T[0]
            #     simobj.T[istep + 1] = t
            #     simobj.Y[istep + 1] = y
            #     x0 = simobj.Y[istep + 1]

            #     # circ.set_center((x0[0], x0[1]))
            #     # self.time_slider.set_val(istep)
            #     # plt.pause(0.001)
            #     # self.fig.canvas.draw_idle()
            #     # self.fig.canvas.flush_events()

            # # keep figure open after finishing
            # # plt.show()

            ######################

            # axis_position = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='white') # type:ignore
            # self.time_slider = Slider(
            #     axis_position,
            #     label='Time (s)',
            #     valmin=0,
            #     valmax=Nt - 1,
            #     valinit=0
            #     )

            self.fig, self.ax = plt.subplots(figsize=(6, 4))
            anim = FuncAnimation(self.fig, self.animate, frames=partial(self.frames, _simobj=simobj), interval=10, blit=False, cache_frame_data=False)
            plt.show()


    def animate(self, frame):
        x, y = frame
        return plt.plot(x, y, color='tab:blue')


    def frames(self, _simobj: SimObject):
        """passes frame data to FuncAnimation"""
        while self.istep < self.Nt - 1:
            self.istep += 1
            self._anim_update(self.istep, _simobj)
            yield (_simobj.T[:self.istep], _simobj.Y[:self.istep])

        # on animate end
        # self.time_slider.on_changed(lambda t: self.update(t, state_slice=state_slice))


    def _anim_update(self, istep: int, _simobj: SimObject) -> None:
        tstep_prev = self.t_array[istep]
        tstep = tstep_prev + self.dt
        x0 = _simobj.Y[istep - 1]

        sol = solve_ivp(
                fun=self.step,
                t_span=(tstep_prev, tstep),
                t_eval=[tstep],
                y0=x0,
                args=(_simobj,),
                events=self.events,
                rtol=1e-3,
                atol=1e-6,
                )
        _simobj.T[istep] = sol['t'][0]
        _simobj.Y[istep] = sol['y'].T[0]

    # def update(self, t: float, state_slice: tuple[int, int]) -> None:
    #     for simobj in self.simobjs:
    #         if isinstance(simobj.patch, patches.Circle):
    #             istep = int(self.time_slider.val)
    #             simobj.patch.set_center(tuple(simobj.Y[istep, state_slice[0]:state_slice[1]]))
    #     self.fig.canvas.draw()

