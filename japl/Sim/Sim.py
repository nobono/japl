# ---------------------------------------------------
from tqdm import tqdm

import numpy as np

from japl.SimObject.SimObject import SimObject

from scipy.integrate import solve_ivp

# ---------------------------------------------------



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

