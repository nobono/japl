from typing import Callable
import numpy as np
from japl import global_opts
from japl.Aero.Atmosphere import Atmosphere
from japl.SimObject.SimObject import SimObject
from japl.DeviceInput.DeviceInput import DeviceInput
from japl.Plotter.Plotter import Plotter
from japl.Plotter.PyQtGraphPlotter import PyQtGraphPlotter
from japl.Sim.Integrate import runge_kutta_4
from japl.Sim.Integrate import euler
from scipy.integrate import solve_ivp
from functools import partial
import time
# import quaternion
# from scipy import constants
# from japl.Library.Vehicles.RigidBodyModel import RigidBodyModel
# from japl.Math import Rotation
# from japl.Math.Vec import vec_ang



class Sim:

    def __init__(self,
                 t_span: list|tuple,
                 dt: float,
                 simobjs: list[SimObject],
                 **kwargs,
                 ) -> None:

        self._dtype = kwargs.get("dtype", np.float64)

        self.t_span = t_span
        self.dt = dt
        self.simobjs = simobjs
        self.events = kwargs.get("events", [])
        self.animate: bool = kwargs.get("animate", False)  # choice of iterating solver over each dt step
        self.integrate_method = kwargs.get("integrate_method", "odeint")
        assert self.integrate_method in ["odeint", "euler", "rk4"]

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
        self.frame_rate: float = kwargs.get("frame_rate", 10)
        self.moving_bounds: bool = kwargs.get("moving_bounds", False)
        self.__instantiate_plot(**kwargs)

        # device inputs
        self.device_input_type = kwargs.get("device_input_type", "")
        self.device_input = DeviceInput(device_type=self.device_input_type)
        self.device_input_data = {"lx": 0.0, "ly": 0.0}

        # atmosphere model
        self.atmosphere = Atmosphere()

        # sim flags
        self.flag_stop = False

        # debug stuff
        # TODO make this its own class so we can use
        # it to profile other classes?
        def _debug_profiler_func():
            if self.debug_profiler["count"] > 1:  # 't' is initally 0, discard this point
                _dt = (time.time() - self.debug_profiler['t'])
                self.debug_profiler["t_total"] += _dt
                self.debug_profiler["t_ave"] = self.debug_profiler["t_total"] / self.debug_profiler["count"]
            self.debug_profiler['t'] = time.time()
            self.debug_profiler["count"] += 1
            if self.debug_profiler["count"] >= self.Nt:
                print("ave_dt: %.5f, ave_Hz: %.1f" % (self.debug_profiler["t_ave"], (1 / self.debug_profiler["t_ave"])))
        self.debug_profiler = {"t": 0.0, "t_total": 0.0, "count": 0, "t_ave": 0.0, "run": _debug_profiler_func}


    def __init_run(self, simobj: SimObject):
        # pre-allocate output arrays
        self.T = np.zeros((self.Nt + 1, ))
        simobj.Y = np.zeros((self.Nt + 1, len(simobj.X0)))
        simobj.Y[0] = simobj.X0
        simobj._set_T_array_ref(self.T)  # simobj.T reference to sim.T


    def __instantiate_plot(self, **kwargs) -> None:
        """This method instantiates the plotter class into the Sim class (if defined).
        Otherwise, a default Plotter class is instantiated."""

        self.plotter = kwargs.get("plotter", None)

        if self.plotter is None:
            if global_opts.get_plotlib() == "matplotlib":
                self.plotter = Plotter(Nt=self.Nt, dt=self.dt, **kwargs)
            elif global_opts.get_plotlib() == "pyqtgraph":
                self.plotter = PyQtGraphPlotter(Nt=self.Nt, dt=self.dt, **kwargs)
            else:
                raise Exception("no Plotter class can be setup.")

            # setup plotter
            self.plotter.setup()

            # add inital simobjs provided
            for simobj in self.simobjs:
                self.plotter.add_simobject(simobj)


    def run(self) -> "Sim":

        # TODO make this better
        simobj = self.simobjs[0]

        # run pre-sim checks
        simobj._pre_sim_checks()

        # begin device input read thread
        if self.device_input_type:
            self.device_input.start()

        # solver
        # TODO must combine all given SimObjects into single state
        if self.animate:
            # solver for one step at a time
            self._solve_with_animation(simobj)
        else:
            # to solve all at once...
            # self._solve(simobj)

            self.__init_run(simobj)

            for istep in range(1, self.Nt + 1):
                self._step_solve(dynamics_func=self.step,
                                 istep=istep,
                                 dt=self.dt,
                                 simobj=simobj,
                                 method=self.integrate_method,
                                 rtol=self.rtol,
                                 atol=self.atol)

        return self


    def _solve_with_animation(self, simobj: SimObject) -> None:
        """This method handles the animation when running the Sim class."""

        self.__init_run(simobj)

        # try to set animation frame intervals to real time
        interval_ms = int(max(1, (1 / self.frame_rate) * 1000))
        step_func = partial(self._step_solve,
                            dynamics_func=self.step,
                            istep=0,
                            dt=self.dt,
                            simobj=simobj,
                            method=self.integrate_method,
                            rtol=self.rtol,
                            atol=self.atol)
        anim_func = partial(self.plotter._animate_func,
                            simobj=simobj,
                            step_func=step_func,
                            frame_rate=interval_ms,
                            moving_bounds=self.moving_bounds)

        anim = self.plotter.FuncAnimation(  # noqa
                func=anim_func,
                frames=self.Nt,
                interval=interval_ms,
                )

        self.plotter.show()


    def step(self, t: float, X: np.ndarray, U: np.ndarray, dt: float, simobj: SimObject):
        """This method is the main step function for the Sim class."""

        ########################################################
        # device input
        ########################################################
        if self.device_input_type:
            iota = -self.device_input_data["ly"] * 0.69  # noqa
        # force = np.array([1000*lx, 0, 1000*ly])
        # acc_ext = acc_ext + force / mass
        ########################################################
        Xdot = simobj.step(X, U, dt)
        return Xdot


    def _step_solve(self,
                    dynamics_func: Callable,
                    istep: int,
                    dt: float,
                    simobj: SimObject,
                    method: str,
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
        -- step_func - function to be integrated
        -- istep - integer step
        -- dt - time step
        -- simobj - SimObject
        -- method - integration method to use
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
        # DEBUG PROFILE #########
        self.debug_profiler["run"]()
        #########################

        # get device input
        if self.device_input_type:
            (lx, ly, _, _) = self.device_input.get()
            self.device_input_data["lx"] = lx
            self.device_input_data["ly"] = ly

        # setup time and initial state for step
        tstep_prev = self.t_array[istep - 1]
        tstep = self.t_array[istep]
        X = simobj.Y[istep - 1]

        # setup input array
        U = np.zeros(len(simobj.model.input_vars), dtype=self._dtype)

        ##################################################################
        # apply direct state updates
        ##################################################################
        # NOTE: avoid overwriting states by using X_temp to
        # process all direct updates before storing values
        # back into X_new.
        ##################################################################

        # apply direct updates to input
        U_temp = simobj.model.direct_input_update_func(tstep, X, U, dt).flatten()
        for i in range(len(U_temp)):
            if not np.isnan(U_temp[i]):
                U[i] = U_temp[i]

        # apply direct updates to state
        X_temp = simobj.model.direct_state_update_func(tstep, X, U, dt).flatten()
        for i in range(len(X_temp)):
            if not np.isnan(X_temp[i]):
                X[i] = X_temp[i]

        ##################################################################
        # Integration Methods
        ##################################################################
        match method:
            case "euler":
                X_new, T_new = euler(
                        f=dynamics_func,
                        t=tstep,
                        X=X,
                        dt=dt,
                        args=(U, dt, simobj,),
                        )
            case "rk4":
                X_new, T_new = runge_kutta_4(
                        f=dynamics_func,
                        t=tstep,
                        X=X,
                        h=dt,
                        args=(U, dt, simobj,),
                        )
            case "odeint":
                sol = solve_ivp(
                        fun=dynamics_func,
                        t_span=(tstep_prev, tstep),
                        t_eval=[tstep],
                        y0=X,
                        args=(U, dt, simobj,),
                        events=self.events,
                        rtol=rtol,
                        atol=atol,
                        max_step=max_step
                        )
                X_new = sol['y'].T[0]
                T_new = sol['t'][0]
            case _:
                raise Exception(f"integration method {self.integrate_method} is not defined")

        # store results
        self.T[istep] = T_new
        simobj.Y[istep] = X_new

        # TODO do this better...
        if self.flag_stop:
            self.plotter.exit()
