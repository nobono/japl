from typing import Callable
import numpy as np
from japl.Aero.Atmosphere import Atmosphere
from japl.SimObject.SimObject import SimObject
from japl.DeviceInput.DeviceInput import DeviceInput
from japl.Sim.Integrate import runge_kutta_4
from japl.Sim.Integrate import euler
from japl.Util.Profiler import Profiler
from scipy.integrate import solve_ivp



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
        self.integrate_method = kwargs.get("integrate_method", "rk4")
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

        # device inputs
        self.device_input_type = kwargs.get("device_input_type", "")
        self.device_input = DeviceInput(device_type=self.device_input_type)
        self.device_input_data = {"lx": 0.0, "ly": 0.0}

        # atmosphere model
        self.atmosphere = Atmosphere()

        # init simobj data arrays
        for simobj in self.simobjs:
            self.__init_simobj(simobj)

        self.profiler = Profiler()


    def __init_simobj(self, simobj: SimObject):
        # pre-allocate output arrays
        self.T = np.zeros((self.Nt + 1, ))
        simobj.Y = np.zeros((self.Nt + 1, len(simobj.X0)))
        simobj.Y[0] = simobj.X0
        simobj._set_T_array_ref(self.T)  # simobj.T reference to sim.T


    def run(self) -> None:

        # TODO: handle multiple simobjs
        simobj = self.simobjs[0]

        # run pre-sim checks
        simobj._pre_sim_checks()

        # begin device input read thread
        if self.device_input_type:
            self.device_input.start()

        # solver
        # TODO should we combine all given SimObjects into single state?
        #       this would be efficient for n-body problem...
        for istep in range(1, self.Nt + 1):
            self._step_solve(dynamics_func=self.step,
                             istep=istep,
                             dt=self.dt,
                             simobj=simobj,
                             method=self.integrate_method,
                             rtol=self.rtol,
                             atol=self.atol)


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
        Xdot = simobj.step(t, X, U, dt)
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
        self.profiler()
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

        # apply any user-defined input functions
        if simobj.model.user_input_function:
            simobj.model.user_input_function(tstep, X, U, dt)

        # apply direct updates to input
        if simobj.model.direct_input_update_func:
            U_temp = simobj.model.direct_input_update_func(tstep, X, U, dt).flatten()
            for i in range(len(U_temp)):
                if not np.isnan(U_temp[i]):
                    U[i] = U_temp[i]

        # apply direct updates to state
        if simobj.model.direct_state_update_func:
            X_temp = simobj.model.direct_state_update_func(tstep, X, U, dt).flatten()
            if X_temp is None:
                raise Exception("Model direct_state_update_func returns None."
                                f"(in SimObject \"{simobj.name})\"")
            for i in range(len(X_temp)):
                if not np.isnan(X_temp[i]):
                    X[i] = X_temp[i]

        if not simobj.model.dynamics_func:
            self.T[istep] = tstep + dt
            simobj.Y[istep] = X
        else:
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
            self.istep += 1
