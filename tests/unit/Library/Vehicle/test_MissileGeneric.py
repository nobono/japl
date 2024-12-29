import os
import unittest
import numpy as np
from astropy import units as u
import quaternion
from japl.SimObject.SimObject import SimObject
from japl.Sim.Sim import Sim
from japl.AeroTable.AeroTable import AeroTable
# from japl import Model
# from sympy import MatrixSymbol, Matrix, symbols
from japl.Library.Vehicles import MissileGeneric
from japl.Sim.Integrate import runge_kutta_4
from japl.Aero.Atmosphere import Atmosphere
from japl.Math import Rotation
from japl.Math import Vec
from japl.global_opts import get_root_dir

ROOT_DIR = get_root_dir()



class test_MissileGeneric(unittest.TestCase):


    def test_MissileGeneric_case1(self):
        simobj = self.create_simobj()
        sim = Sim(
                t_span=self.t_span,
                dt=self.dt,
                simobjs=[simobj],
                integrate_method="rk4",
                animate=0,
                quiet=1,
                )
        sim.run()

        # truth = self.run_dynamics()

        # debug printing
        # state_id = -2
        # fmt = "%.16f"
        # rng = (-10, 0)
        # print()
        # for i in range(*rng):
        #     print(f"symb: {fmt}" % simobj.Y[i, state_id])
        # print()
        # for i in range(*rng):
        #     print(f"func: {fmt}" % truth[i, state_id])
        # print()
        # print(truth[-1] - simobj.Y[-1])


        # TEMP #####################################
        # NOTE: this is very specific testing case

        # for i in simobj.Y[-1]:
        #     print("%.18f," % i)

        # tru = [
        #     150.000000000000028422,
        #     0.000000000000000000,
        #     10000.048684187155231484,
        #     1500.000000000000000000,
        #     0.000000000000000000,
        #     0.961120785433739355,
        #     0.000000000000000000,
        #     -0.000282937163954947,
        #     0.000000000000000000,
        #     0.999999999782131499,
        #     0.000000000000000000,
        #     0.000020874318832341,
        #     0.000000000000000000,
        #     133.000000000000000000,
        #     1.419999999999999929,
        #     1.308999999999999941,
        #     58.270000000000003126,
        #     58.270000000000003126,
        #     9.775868442887434284,
        #     1500.000251441063483071,
        #     5.007818706472199288,
        #     -0.000536139293023611,
        #     0.000000000000000000]

        # for t, y in zip(tru, simobj.Y[-1]):
        #     self.assertAlmostEqual(t, y, places=16)
        ############################################

        # for state, tru in zip(simobj.Y[-1], truth[-1]):
        #     self.assertAlmostEqual(state, tru, places=self.TOLERANCE_PLACES)


    def setUp(self):
        self.TOLERANCE_PLACES = 15
        self._dtype = float
        self.dt = 0.01
        self.t_span = [0, 0.1]
        self.atmosphere = Atmosphere()
        self.aerotable = AeroTable(f"{ROOT_DIR}/aerodata/aeromodel_psb.mat",
                                   # keep_units=True,
                                   # angle_units=u.deg,
                                   # length_units=u.imperial.foot
                                   )
        # self.aerotable = AeroTable("../../../aerodata/aeromodel_psb.mat")


    def create_simobj(self):
        model = MissileGeneric.model
        simobj = SimObject(model=model, size=2, color='tab:blue', dtype=self._dtype)
        # simobj.Ixx = 1.309  # (kg * m^2)
        # simobj.Iyy = 58.27  # (kg * m^2)
        # simobj.Izz = 58.27  # (kg * m^2)
        # simobj.mass = 133  # (kg)
        # simobj.cg = 1.42  # (m)
        x0 = [0, 0, 10000]
        v0 = [1500, 0, 0]
        w0 = [0, 0, 0]
        quat0 = quaternion.from_euler_angles([0, 0, 0]).components
        mass0 = 133.0
        cg0 = 1.42
        Ixx0 = 1.309
        Iyy0 = 58.27
        Izz0 = 58.27
        gacc0 = -9.81
        speed0 = np.linalg.norm(v0)
        mach0 = speed0 / self.atmosphere.speed_of_sound(x0[2])
        alpha0 = 0
        phi0 = 0

        simobj.init_state([x0,
                           v0,
                           w0,
                           quat0,
                           mass0,
                           cg0,
                           Ixx0,
                           Iyy0,
                           Izz0,
                           gacc0,
                           speed0,
                           mach0,
                           alpha0,
                           phi0,
                           ], dtype=self._dtype)

        return simobj


    def dynamics(self, t, X, U, dt, simobj):
        # pos = X[:3]
        vel = X[3:6]
        angvel = X[6:9]
        quat = X[9:13]
        mass = X[13]
        # cg = X[14]
        Ixx = X[15]
        Iyy = X[16]
        Izz = X[17]
        gacc = X[18]
        # speed = X[19]
        # mach = X[20]
        # alpha = X[21]
        # phi = X[22]

        force = U[:3]
        torque = U[3:6]

        acc = force / mass
        angacc = np.array([torque[0] / Ixx,
                           torque[1] / Iyy,
                           torque[2] / Izz], dtype=self._dtype)

        gravity = np.array([0, 0, gacc])

        wx, wy, wz = angvel
        Sw = np.array([
            [0, wx, wy, wz],  # type:ignore
            [-wx, 0, -wz, wy],  # type:ignore
            [-wy, wz, 0, -wx],  # type:ignore
            [-wz, -wy, wx, 0],  # type:ignore
            ], dtype=self._dtype)

        pos_dot = vel
        vel_dot = acc + gravity
        angvel_dot = angacc
        quat_dot = -(0.5 * Sw @ quat)
        mass_dot = 0
        cg_dot = 0
        Ixx_dot = 0
        Iyy_dot = 0
        Izz_dot = 0
        gacc_dot = 0
        speed_dot = 0
        mach_dot = 0
        alpha_dot = 0
        phi_dot = 0

        Xdot = np.array([
            *pos_dot,
            *vel_dot,
            *angvel_dot,
            *quat_dot,
            mass_dot,
            cg_dot,
            Ixx_dot,
            Iyy_dot,
            Izz_dot,
            gacc_dot,
            speed_dot,
            mach_dot,
            alpha_dot,
            phi_dot,
            ], dtype=self._dtype)

        return Xdot


    def direct_updates(self, X, U, dt, simobj: SimObject):
        pos = X[:3]
        vel = X[3:6]
        # angvel = X[6:9]
        # quat = X[9:13]
        # mass = X[13]
        # cg = X[14]
        # Ixx = X[15]
        # Iyy = X[16]
        # Izz = X[17]
        # gacc = X[18]
        # speed = X[19]
        # mach = X[20]
        # alpha = X[21]
        # phi = X[22]

        force = U[:3]
        torque = U[3:6]

        alt = pos[2]
        gacc_new = self.atmosphere.grav_accel(alt)
        sos_new = self.atmosphere.speed_of_sound(alt)
        speed_new = np.linalg.norm(vel)
        mach_new = speed_new / sos_new

        (alpha_new,
         phi_new,
         force_aero,
         torque_aero) = self.aerotable_update(X)

        force_new = force + force_aero
        torque_new = torque + torque_aero

        X[18] = gacc_new
        X[19] = speed_new
        X[20] = mach_new
        X[21] = alpha_new
        X[22] = phi_new

        U[:3] = force_new
        U[3:6] = torque_new

        return (X, U)


    def aerotable_update(self, X):
        pos = X[:3]
        vel = X[3:6]
        quat = X[9:13]
        cg = X[14]
        Iyy = X[16]
        speed = X[19]
        mach = X[20]
        alpha = X[21]
        phi = X[22]

        alt = pos[2]

        # calc angle of attack: (pitch_angle - flight_path_angle)
        vel_hat = vel / speed

        # projection vel_hat --> x-axis
        zx_plane_norm = np.array([0, 1, 0], dtype=self._dtype)
        vel_hat_zx = ((vel_hat @ zx_plane_norm) / np.linalg.norm(zx_plane_norm)) * zx_plane_norm
        vel_hat_proj = vel_hat - vel_hat_zx

        # get Trait-bryan angles (yaw, pitch, roll)
        yaw_angle, pitch_angle, roll_angle = Rotation.quat_to_tait_bryan(np.asarray(quat), dtype=self._dtype)

        # angle between proj vel_hat & xaxis
        x_axis_inertial = np.array([1, 0, 0], dtype=self._dtype)
        ang = Vec.vec_ang(vel_hat_proj, x_axis_inertial)

        flight_path_angle = np.sign(vel_hat_proj[2]) * ang
        alpha_new = pitch_angle - flight_path_angle                     # angle of attack
        phi_new = roll_angle

        iota = np.radians(0.1)
        CLMB = -self.aerotable.get_CLMB(alpha=alpha, phi=phi, mach=mach, iota=iota)  # type:ignore
        CNB = self.aerotable.get_CNB(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)  # type:ignore
        My_coef = CLMB + (cg - self.aerotable.get_MRC()) * CNB  # type:ignore

        q = self.atmosphere.dynamic_pressure(vel, alt)  # type:ignore
        Sref = self.aerotable.get_Sref()
        Lref = self.aerotable.get_Lref()
        My = My_coef * q * Sref * Lref

        force_z_aero = CNB * q * Sref  # type:ignore
        force_aero = np.array([0, 0, force_z_aero], dtype=self._dtype)

        torque_y_new = My / Iyy
        torque_aero = np.array([0, torque_y_new, 0], dtype=self._dtype)

        return (alpha_new, phi_new, force_aero, torque_aero)


    def run_dynamics(self):
        Nt = int(self.t_span[1] / self.dt)
        t_array = np.linspace(self.t_span[0], self.t_span[1], Nt + 1)
        simobj = self.create_simobj()
        simobj.Y = np.zeros((Nt + 1, len(simobj.X0)), dtype=self._dtype)
        simobj.Y[0] = simobj.X0

        for istep in range(1, Nt + 1):
            tstep = t_array[istep]
            X = simobj.Y[istep - 1]
            U = np.zeros(len(simobj.model.input_vars))
            args = (U, self.dt, simobj)

            # direct update for input
            X, U = self.direct_updates(X, U, self.dt, simobj)

            X_new, T_new = runge_kutta_4(
                    f=self.dynamics,
                    t=tstep,
                    X=X,
                    dt=self.dt,
                    args=args
                    )

            simobj.Y[istep] = X_new
        truth = simobj.Y
        return truth


if __name__ == '__main__':
    unittest.main()
