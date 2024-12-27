from typing import Callable
import numpy as np
from japl.Aero.Atmosphere import Atmosphere
from japl.SimObject.SimObject import SimObject
from japl.DeviceInput.DeviceInput import DeviceInput
from japl.Sim.Integrate import runge_kutta_4
from japl.Sim.Integrate import euler
from japl.Util.Profiler import Profiler
from scipy.integrate import solve_ivp


# NOTE below is currently unused feature ----------------------------------
# which may be used in the future.
#
# ########################################################
# # device input
# ########################################################
# if self.device_input_type:
#     iota = -self.device_input_data["ly"] * 0.69  # noqa
# # force = np.array([1000*lx, 0, 1000*ly])
# # acc_ext = acc_ext + force / mass
# ########################################################
#
# # get device input
# if self.device_input_type:
#     (lx, ly, _, _) = self.device_input.get()
#     self.device_input_data["lx"] = lx
#     self.device_input_data["ly"] = ly
# -------------------------------------------------------------------------


class Sim:

    """This class configures a Sim object which will run SimObject(s)
    and the underlying Models."""

    def __init__(self,
                 t_span: list|tuple,
                 dt: float,
                 simobjs: list[SimObject],
                 **kwargs,
                 ) -> None:
        """
        -------------------------------------------------------------------

        Parameters:
            t_span: time span [low, high] to run simulation

            dt: time increment

            simobjs: list of SimObjects to run within the simulation


        -------------------------------------------------------------------
        """

        self._dtype = kwargs.get("dtype", np.float64)

        self.t_span = t_span
        self.dt = dt
        self.simobjs = simobjs
        self.events: list[tuple] = kwargs.get("events", [])
        self.integrate_method = kwargs.get("integrate_method", "rk4")
        if self.integrate_method not in ["odeint", "euler", "rk4"]:
            raise Exception(f"integrate method: {self.integrate_method} not recongized. "
                            "only [odeint, euler, rk4] available.")

        # setup time array
        self.istep = 0
        self.Nt = int(self.t_span[1] / self.dt)
        self.t_array = np.linspace(self.t_span[0], self.t_span[1], self.Nt + 1)
        self.T = np.zeros((self.Nt + 1,))

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
            for child in simobj.children_pre_update:
                child._init_data_array(self.T)
            simobj._init_data_array(self.T)
            for child in simobj.children_post_update:
                child._init_data_array(self.T)

        self.profiler = Profiler()


    def add_event(self, func: Callable, action: str) -> None:
        """Adds an event to the simulation"""
        # TODO make better...
        self.events += [(action, func)]


    def run(self) -> None:
        """Runs the simulation"""

        # TODO: handle multiple simobjs
        simobj = self.simobjs[0]

        # run pre-sim checks
        simobj._pre_sim_checks()

        # begin device input read thread
        if self.device_input_type:
            self.device_input.start()

        for istep in range(1, self.Nt + 1):
            self.istep = istep
            self._run(istep=istep, simobj=simobj)

        # OLD METHOD ----------------------------------------------------------
        # # solver
        # # TODO should we combine all given SimObjects into single state?
        # #       this would be efficient for n-body problem...
        # for istep in range(1, self.Nt + 1):
        #     flag_sim_stop = self.step_solve(dynamics_func=self.step,
        #                                     istep=istep,
        #                                     dt=self.dt,
        #                                     T=self.T,
        #                                     t_array=self.t_array,
        #                                     simobj=simobj,
        #                                     method=self.integrate_method,
        #                                     events=self.events,
        #                                     rtol=self.rtol,
        #                                     atol=self.atol)
        #     if flag_sim_stop:
        #         break
        # ---------------------------------------------------------------------


    def _run(self, istep: int, simobj: SimObject) -> bool:
        # update mixing ---------------------------------------------------
        # updates from input, state, dynamics need to be kept separate
        # then mixed correctly (NOTE this is a certainty?).
        # -----------------------------------------------------------------
        for child in simobj.children_pre_update:
            flag_sim_stop = self._run(istep=istep, simobj=child)
            if flag_sim_stop:
                return flag_sim_stop

        # update SimObject time step index
        simobj.set_istep(istep)

        flag_sim_stop = self.step_solve(istep=istep, simobj=simobj, dt=self.dt, T=self.T,
                                        t_array=self.t_array, method=self.integrate_method,
                                        events=self.events, rtol=self.rtol, atol=self.atol)
        if flag_sim_stop:
            return flag_sim_stop

        for child in simobj.children_post_update:
            flag_sim_stop = self._run(istep=istep, simobj=child)
            if flag_sim_stop:
                return flag_sim_stop

        return False


    @staticmethod
    def step_solve(istep: int,
                   dt: float,
                   T: np.ndarray,
                   t_array: np.ndarray,
                   simobj: SimObject,
                   method: str,
                   events: list[tuple],
                   rtol: float = 1e-6,
                   atol: float = 1e-6,
                   max_step: float = 0.2
                   ) -> bool:
        """
            This method is an update step for the ODE solver from time step 't' to 't + dt';
        used by FuncAnimation.

        -------------------------------------------------------------------

        Parameters:
            istep: integer step
            dt: time step
            T: simulation time array
            t_array: simulation time-step array
            simobj: SimObject being processed
            method: integration method to use
            rtol: relative tolerance for ODE Solver
            atol: absolute tolerance for ODE Solver
            max_step: max step size for ODE Solver (seconds)

        Returns:
            sim_stop_flag: flag to stop simulation. based on event trigger or
                           end of sim.
            t: time array of solution points
            y: (Nt x N_state) array of solution points from ODE solver

        -------------------------------------------------------------------
        """
        flag_sim_stop = False

        # DEBUG PROFILE #########
        # self.profiler()
        #########################

        # setup time and initial state for step
        tstep = t_array[istep - 1]
        tstep_next = t_array[istep]
        X = simobj.Y[istep - 1].copy()  # init with previous state
        U = simobj.U[istep - 1].copy()      # init with current input array (zeros)
        S = simobj.S0

        ##################################################################
        # apply direct state updates
        ##################################################################
        # NOTE: avoid overwriting states by using X_temp to
        # process all direct updates before storing values
        # back into X_new.
        ##################################################################

        # TODO: (working) expanding for matrix
        # state_mat_reshape_info = []
        # for name, info in simobj.model.state_register.matrix_info.items():
        #     state_mat_reshape_info += [(info["id"], info["size"], info["var"].shape)]

        X_prev = X.copy()
        X_state_update = np.empty_like(X)
        X_state_update[:] = np.nan

        # -----------------------------------------------------------
        # run user-defined functions here, before parent SimObject's
        # model update step.
        # TODO THIS IS NOT TESTED
        # for func in simobj.model.pre_update_functions:
        #     func(tstep, X_prev.copy(), U.copy(), S, dt, simobj)
        # -----------------------------------------------------------

        # apply any user-defined input functions
        if simobj.model.has_input_function():
            U = simobj.model.input_function(tstep, X_prev, U.copy(), S, dt, simobj)

        # apply direct updates to input
        if simobj.model.has_input_updates():
            # TODO: (working) expanding for matrix
            # for info in state_mat_reshape_info:
            #     id, size, shape = info
            U_temp = simobj.model.input_updates(tstep, X.copy(), U, S, dt).flatten()
            input_update_mask = ~np.isnan(U_temp)
            U[input_update_mask] = U_temp[input_update_mask]  # ignore nan values

        # apply direct updates to state
        if simobj.model.has_state_updates():
            X_state_update = simobj.model.state_updates(tstep, X_prev, U, S, dt).flatten()
            if X_state_update is None:
                raise Exception("Model direct_state_update_func returns None."
                                f"(in SimObject \"{simobj.name})\"")

        if not simobj.model.has_dynamics():
            T[istep] = tstep + dt
            simobj.Y[istep] = X
            simobj.U[istep] = U
        else:
            ##################################################################
            # Integration Methods
            ##################################################################
            match method:
                case "euler":
                    X_new, T_new = euler(
                            f=simobj.model.dynamics,
                            t=tstep,
                            X=X_prev,
                            dt=dt,
                            args=(U, S, dt, simobj,),
                            )
                    # mix non-NaN values from dynamics
                    # and direct / external updates
                    mask = ~np.isnan(X_state_update)
                    X_new[mask] = X_state_update[mask]

                case "rk4":
                    X_new, T_new = runge_kutta_4(
                            f=simobj.model.dynamics,
                            t=tstep,
                            X=X_prev,
                            dt=dt,
                            args=(U, S, dt,),
                            # args=(U, S, dt, simobj,),
                            )
                    # mix non-NaN values from dynamics
                    # and direct / external updates
                    mask = ~np.isnan(X_state_update)
                    X_new[mask] = X_state_update[mask]

                case "odeint":
                    sol = solve_ivp(
                            fun=simobj.model.dynamics,
                            t_span=(tstep, tstep_next),
                            t_eval=[tstep_next],
                            y0=X,
                            args=(U, S, dt, simobj,),
                            events=events,
                            rtol=rtol,
                            atol=atol,
                            max_step=max_step
                            )
                    X_new = sol['y'].T[0]
                    T_new = sol['t'][0]
                    # mix non-NaN values from dynamics
                    # and direct / external updates
                    mask = ~np.isnan(X_state_update)
                    X_new[mask] = X_state_update[mask]

                case _:
                    raise Exception(f"integration method {method} is not defined")

            # run user-defined functions here, after parent SimObject's
            # model update step.
            for func in simobj.model.post_update_functions:
                func(tstep, X_new.copy(), U.copy(), S, dt, simobj)

            # store results
            T[istep] = T_new
            simobj.Y[istep] = X_new
            simobj.U[istep] = U
            istep += 1

            # check events
            for event in events:
                action, event_func = event
                flag_event = event_func(tstep, X, U, S, dt, simobj)
                if flag_event:
                    match action:
                        case "stop":
                            # trim output arrays
                            T = T[:istep + 1]
                            simobj.Y = simobj.Y[:istep + 1]
                            simobj.U = simobj.U[:istep + 1]
                            simobj._set_T_array_ref(T)
                            flag_sim_stop = True
                        case _:
                            pass
            return flag_sim_stop
        return flag_sim_stop
