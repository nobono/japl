import numpy as np
import quaternion

from japl import global_opts
from japl.Aero.Atmosphere import Atmosphere
from japl.Math.Rotation import quat_to_tait_bryan
from japl.Math.Vec import vec_ang
from japl.SimObject.SimObject import SimObject
from japl.DeviceInput.DeviceInput import DeviceInput
from japl.Plotter.Plotter import Plotter
from japl.Plotter.PyQtGraphPlotter import PyQtGraphPlotter

from japl.Sim.Integrate import runge_kutta_4

from scipy.integrate import solve_ivp

from functools import partial

from scipy import constants

import time

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
            if self.debug_profiler["count"] > 1: # 't' is initally 0, discard this point
                _dt = (time.time() - self.debug_profiler['t'])
                self.debug_profiler["t_total"] += _dt
                self.debug_profiler["t_ave"] = self.debug_profiler["t_total"] / self.debug_profiler["count"]
            self.debug_profiler['t'] = time.time()
            self.debug_profiler["count"] += 1
            if self.debug_profiler["count"] >= self.Nt:
                print("ave_dt: %.5f, ave_Hz: %.1f" % (self.debug_profiler["t_ave"], (1 / self.debug_profiler["t_ave"])))
        self.debug_profiler = {"t": 0.0, "t_total": 0.0, "count": 0, "t_ave": 0.0, "run": _debug_profiler_func}


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
        if not self.animate:
            # TODO must combine all given SimObjects into single state
            # to solve all at once...
            self.solve(simobj)

        # solver for one step at a time
        elif self.animate:
            self.solve_with_animation(simobj)

        return self


    def solve(self, simobj: SimObject) -> None:
        """This method handles running the Sim class using an ODE Solver"""

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

        # TODO handle visualization afterwards...

        # xdata, ydata = simobj.get_plot_data()
        # simobj._update_patch_data(xdata, ydata)

        # self.plotter.autoscale(xdata, ydata)
        # self.plotter.setup_time_slider(self.Nt, [simobj])

        self.plotter.show()


    def solve_with_animation(self, simobj: SimObject) -> None:
        """This method handles the animation when running the Sim class."""

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


    def step(self, t, X, simobj: SimObject):
        """This method is the main step function for the Sim class."""

        # TODO make "ac" automatically the correct length
        acc_ext = np.array([0, 0, 0], dtype=float)
        torque_ext = np.array([0, 0, 0], dtype=float)

        mass = X[simobj.get_state_id("mass")]

        iota = np.radians(0.1)

        # device input
        if self.device_input_type:
            iota = -self.device_input_data["ly"] * 0.69
        # force = np.array([1000*lx, 0, 1000*ly])
        # acc_ext = acc_ext + force / mass

        ########################################################################
        # Aeromodel
        ########################################################################
        if simobj.aerotable:
            # get current states
            alt = X[simobj.get_state_id("z")]
            vel = X[simobj.get_state_id(["vx", "vy", "vz"])]
            quat = X[simobj.model.get_state_id(["q0", "q1", "q2", "q3"])]

            # calculate current mach
            speed = float(np.linalg.norm(vel))
            mach = (speed / self.atmosphere.speed_of_sound(alt))

            # calc angle of attack: (pitch_angle - flight_path_angle)
            vel_hat = vel / speed                                       # flight path vector

            # projection vel_hat --> x-axis
            zx_plane_norm = np.array([0, 1, 0])
            vel_hat_zx = ((vel_hat @ zx_plane_norm) / np.linalg.norm(zx_plane_norm)) * zx_plane_norm
            vel_hat_proj = vel_hat - vel_hat_zx

            # get Trait-bryan angles (yaw, pitch, roll)
            yaw_angle, pitch_angle, roll_angle = quat_to_tait_bryan(np.asarray(quat))

            # angle between proj vel_hat & xaxis
            x_axis_intertial = np.array([1, 0, 0])
            flight_path_angle = np.sign(vel_hat_proj[2]) * vec_ang(vel_hat_proj, x_axis_intertial)
            alpha = pitch_angle - flight_path_angle                     # angle of attack
            phi = roll_angle

            ###################################################################
            # pitching moment coeficient for iota increments
            # My_coef = model.CLM(alpha,mach)+...
            #           model.cmit(alpha*n,mach*n,model.increments.iota)+...
            #          ((cg-model.MRC)/model.lref)*...
            #          (model.CN(alpha,mach)+...
            #           model.CNit(alpha*n,mach*n,model.increments.iota));

            # % get dynamic pressure (N/m^2)
            # qbar = (model.conv.lbf2n/model.conv.ft2m^2)*...
            #         computeqbar(mach,model.conv.m2ft*alt);

            # Fvec = (qbar*model.sref)*...
            #        ([-model.CA_Basic(alpha,mach)+...
            #          -model.CA0(mach,alt,boost);...
            #          -model.CN(alpha,mach)]+...% basic stability
            #         [-model.cadit(alpha, mach, iota);...
            #          -model.CNit(alpha, mach, iota)]);% b-plane steering
            #
            # My   = (qbar*model.sref*model.lref)*...
            #        (model.CLM(alpha, mach)+...       % basic stability
            #         model.cmit(alpha, mach, iota));   % b-plane steering
            ###################################################################

            # lookup coefficients
            try:
                CLMB = -simobj.aerotable.get_CLMB_Total(alpha, phi, mach, iota)
                CNB = simobj.aerotable.get_CNB_Total(alpha, phi, mach, iota)

                My_coef = CLMB + (simobj.cg - simobj.aerotable.MRC[0]) * CNB

                # calulate moments: (M_coef * q * Sref * Lref), where:
                #       M_coef - moment coeficient
                #       q      - dynamic pressure
                #       Sref   - surface area reference (wing area)
                #       Lref   - length reference (mean aerodynamic chord)
                q = self.atmosphere.dynamic_pressure(vel, alt)
                My = My_coef * q * simobj.aerotable.Sref * simobj.aerotable.Lref
                zforce = CNB * q * simobj.aerotable.Sref

                # update external moments
                # (positive )
                torque_ext[1] = My / simobj.Iyy
                acc_ext[2] = acc_ext[2] + zforce / mass
                # print(torque_ext[1], acc_ext[2], q)

            except Exception as e:
                print(e)
                self.flag_stop = True

        ########################################################################

        U = np.concatenate([acc_ext, torque_ext])
        Xdot = simobj.step(X, U)

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
        # DEBUG PROFILE #########
        self.debug_profiler["run"]()
        #########################

        # get device input
        if self.device_input_type:
            (lx, ly, _, _) = self.device_input.get()
            self.device_input_data["lx"] = lx
            self.device_input_data["ly"] = ly

        tstep_prev = self.t_array[istep - 1]
        tstep = self.t_array[istep]
        x0 = _simobj.Y[istep - 1]

        if self.integrate_method == "rk4":
            X_new, T_new = runge_kutta_4(
                    f=self.step,
                    t=tstep,
                    X=x0,
                    h=self.dt,
                    args=(_simobj,),
                    )
            self.T[istep] = T_new
            _simobj.Y[istep] = X_new

        elif self.integrate_method == "odeint":
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

        else:
            raise Exception(f"integration method {self.integrate_method} is not defined")

        # TODO do this better...
        if self.flag_stop:
            self.plotter.exit()

