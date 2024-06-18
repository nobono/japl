import numpy as np

from japl.Aero.Atmosphere import Atmosphere
from japl.Math.Vec import vec_ang
from japl.SimObject.SimObject import SimObject
from japl.DeviceInput.DeviceInput import DeviceInput
from japl.Plotter.Plotter import Plotter
from japl.Plotter.PyQtGraphPlotter import PyQtGraphPlotter

from japl.Aero import AeroTable

from scipy.integrate import solve_ivp

from functools import partial

from scipy import constants

import japl

# ---------------------------------------------------



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
        aspect: float|str = kwargs.get("aspect", "equal")
        blit: bool = kwargs.get("blit", False)
        cache_frame_data: bool = kwargs.get("cache_frame_data", False)
        repeat: bool = kwargs.get("repeat", False)
        antialias: float = kwargs.get("antialias", False)
        figsize: tuple = kwargs.get("figsize", (6, 4))
        self.moving_bounds: bool = kwargs.get("moving_bounds", False)

        if japl.get_plotlib() == "matplotlib":
            self.plotter = kwargs.get("plotter", Plotter(Nt=self.Nt,
                                                         blit=blit,
                                                         cache_frame_data=cache_frame_data,
                                                         repeat=repeat,
                                                         aspect=aspect,
                                                         antialias=antialias,
                                                         figsize=figsize,
                                                         )
                                      )
        elif japl.get_plotlib() == "pyqtgraph":
            self.plotter = kwargs.get("plotter", PyQtGraphPlotter(Nt=self.Nt,
                                                         blit=blit,
                                                         cache_frame_data=cache_frame_data,
                                                         repeat=repeat,
                                                         aspect=aspect,
                                                         antialias=antialias,
                                                         figsize=figsize,
                                                         )
                                      )

        # device inputs
        self.device_input_type = kwargs.get("device_input_type", "")
        self.device_input = DeviceInput(device_type=self.device_input_type)

        # atmosphere model
        self.atmosphere = Atmosphere()


    def run(self) -> "Sim":

        # TODO make this better
        simobj = self.simobjs[0]

        # begin device input read thread
        if self.device_input_type:
            self.device_input.start()

        self.plotter.setup(self.simobjs)


        ################################
        # solver
        ################################
        if not self.animate:
            # TODO must combine all given SimObjects into single state
            # to solve all at once...

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

            # xdata, ydata = simobj.get_plot_data()
            # simobj._update_patch_data(xdata, ydata)

            # self.plotter.autoscale(xdata, ydata)
            # self.plotter.setup_time_slider(self.Nt, [simobj])

        ################################
        # solver for one step at a time
        ################################
        elif self.animate:

            # pre-allocate output arrays
            self.T = np.zeros((self.Nt + 1, ))
            simobj.Y = np.zeros((self.Nt + 1, len(simobj.X0)))
            simobj.Y[0] = simobj.X0
            simobj._set_T_array_ref(self.T) # simobj.T reference to sim.T

            # try to set animation frame intervals to real time
            interval = int(max(1, self.dt * 1000))
            _step_func = partial(self._step_solve_ivp, istep=0, _simobj=simobj, rtol=self.rtol, atol=self.atol)
            _anim_func = partial(self.plotter._animate_func, _simobj=simobj, step_func=_step_func, moving_bounds=self.moving_bounds)

            anim = self.plotter.FuncAnimation(
                    func=_anim_func,
                    frames=self.Nt,
                    interval=interval,
                    )

        self.plotter.show()

        return self


    def step(self, t, X, simobj: SimObject):

        ac = np.array([0, 0, -constants.g])

        # get device input
        if self.device_input_type:
            (lx, ly, _, _) = self.device_input.get()
            ac = ac + np.array([100*lx, 0, 100*ly])

        #############
        # Aeromodel
        #############
        # inertia_inv = np.array([
        #     [1/X[simobj.Ixx], 0, 0],
        #     [0, 1/X[simobj.Iyy], 0],
        #     [0, 0, 1/X[simobj.Izz]],
        #     ])
        if simobj.aerotable:
            vel = X[simobj.get_state_id("vx"):simobj.get_state_id("vz") + 1]
            speed = float(np.linalg.norm(vel))
            vel_hat = vel / speed
            xaxis = np.array([1, 0, 0])
            alpha = vec_ang(vel_hat, xaxis)     # angle-of-attack
            phi = 0
            alt = X[simobj.get_state_id("z")]
            mach = (speed / self.atmosphere.speed_of_sound(alt))
            iota = 0

            CLMB = simobj.aerotable.get_CLMB_Total(alpha, phi, mach, iota)
            CNB = simobj.aerotable.get_CNB_Total(alpha, phi, mach, iota)

            ytorque = CLMB / simobj.Iyy
            zforce = CNB
            pass
        #############

        # fuel_burn = X[6]
        # if fuel_burn < 100:
        #     ac[0] += 20
        #     ac[2] += 40

        burn_const = 0.4

        U = np.array([*ac])
        Xdot = simobj.step(X, U)
        Xdot[6] = burn_const * np.linalg.norm(ac) #type:ignore

        return Xdot


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


