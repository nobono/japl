# ---------------------------------------------------
from typing import Callable

from tqdm import tqdm

import numpy as np

from japl.SimObject.SimObject import SimObject

from scipy.integrate import solve_ivp

# ---------------------------------------------------

from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
import time



class Sim:

    def __init__(self,
                 t_span: list|tuple,
                 dt: float,
                 simobjs: list[SimObject],
                 events: list = [],
                 step_solve: bool = False,
                 ) -> None:
        self.t_span = t_span
        self.dt = dt
        self.simobjs = simobjs
        self.events = events
        self.step_solve = step_solve # choice of iterating solver over each dt step


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

        # setup time array
        Nt = int(self.t_span[1] / self.dt)
        t_array = np.linspace(self.t_span[0], self.t_span[1], Nt)

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

        if not self.step_solve:
            sol = solve_ivp(
                    fun=self.step,
                    t_span=self.t_span,
                    t_eval=t_array,
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

        elif self.step_solve:

            simobj.T = np.zeros((Nt, ))
            simobj.Y = np.zeros((Nt, len(simobj.X0)))
            x0 = simobj.X0

            ####################################
            # TEMP
            ####################################
            self.fig, self.ax = plt.subplots(figsize=(6, 4))

            axis_position = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='white') # type:ignore
            self.time_slider = Slider(
                axis_position,
                label='Time (s)',
                valmin=0,
                valmax=Nt - 1,
                valinit=0
                )

            state_slice = (0, 2) # user choice of states to plot
            # self.time_slider.on_changed(lambda t: self.update(t, state_slice=state_slice))
            plt.show(block=False)
            self.ax.add_patch(simobj.patch)

            # FuncAnimation
            # def anim_update(args):
            #     return plt.scatter(0, args)

            # def frames():
            #     while True:
            #         yield regr_magic()

            ####################################

            for istep, (tstep_prev, tstep) in tqdm(enumerate(zip(t_array, t_array[1:])), total=Nt):

                sol = solve_ivp(
                        fun=self.step,
                        t_span=(tstep_prev, tstep),
                        t_eval=[tstep],
                        y0=x0,
                        args=(simobj,),
                        events=self.events,
                        rtol=1e-3,
                        atol=1e-6,
                        )

                # check for stop event
                # if check_for_events(sol['t_events']):
                #     # truncate output arrays if early stoppage
                #     T = T[:istep + 1]
                #     Y = Y[:istep + 1]
                #     break
                # else:
                # store output
                t = sol['t'][0]
                y = sol['y'].T[0]
                simobj.T[istep + 1] = t
                simobj.Y[istep + 1] = y
                x0 = simobj.Y[istep + 1]

                # circ.set_center((x0[0], x0[1]))
                # self.time_slider.set_val(istep)
                # plt.pause(0.001)
                # self.fig.canvas.draw_idle()
                # self.fig.canvas.flush_events()

            # keep figure open after finishing
            # plt.show()

            ######################
            count = -1
            xx0 = simobj.Y[0, :]

            # def frames():
            #     nonlocal count
            #     while True:
            #         count += 1
            #         yield count

            def uupdate(istep) -> tuple:
                nonlocal xx0
                tstep_prev = t_array[istep]
                tstep = t_array[istep + 1]
                sol = solve_ivp(
                        fun=self.step,
                        t_span=(tstep_prev, tstep),
                        t_eval=[tstep],
                        y0=xx0,
                        args=(simobj,),
                        events=self.events,
                        rtol=1e-3,
                        atol=1e-6,
                        )
                t = sol['t'][0]
                y = sol['y'].T[0]
                xx0 = y
                # return (self.simobjs[0].patch.set_center((y[0], y[1])), )
                return y[0], y[1]

            # anim = FuncAnimation(self.fig, uupdate, frames=frames, interval=100)
            # plt.show()

            ####################

            class RegrMagic(object):
                """Mock for function Regr_magic()
                """
                def __init__(self):
                    self.x = 0
                def __call__(self):
                    # time.sleep(np.random.random())
                    self.x += 1
                    ret = uupdate(self.x)
                    return ret[0], ret[1]
                    # return self.x, np.random.random()

            regr_magic = RegrMagic()

            def frames():
                nonlocal count
                while True:
                    yield regr_magic()

            fig = plt.figure()

            x = []
            y = []
            def animate(args):
                x.append(args[0])
                y.append(args[1])
                return plt.plot(x, y, color='g')


            anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)
            plt.show()


    def update(self, t: float, state_slice: tuple[int, int]) -> None:
        for simobj in self.simobjs:
            if isinstance(simobj.patch, patches.Circle):
                istep = int(self.time_slider.val)
                simobj.patch.set_center(tuple(simobj.Y[istep, state_slice[0]:state_slice[1]]))
        self.fig.canvas.draw()

