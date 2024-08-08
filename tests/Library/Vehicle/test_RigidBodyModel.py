import unittest
import numpy as np
import quaternion
from japl import Model
from japl import SimObject
from japl import Sim
from japl import AeroTable
from sympy import MatrixSymbol, Matrix, symbols
from japl.Library.Vehicles import RigidBodyModel
from japl.Sim.Integrate import runge_kutta_4



class test_RigidBodyModel(unittest.TestCase):


    def test_RigidBodyModel_case1(self):
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

        truth = self.run_dynamics()

        for state, tru in zip(simobj.Y[-1], truth[-1]):
            self.assertEqual(state, tru)


    def setUp(self):
        self.TOLERANCE_PLACES = 14


    def create_simobj(self):
        model = RigidBodyModel.model
        simobj = SimObject(model=model, size=2, color='tab:blue')
        simobj.Ixx = 1.309 # (kg * m^2)
        simobj.Iyy = 58.27 # (kg * m^2)
        simobj.Izz = 58.27 # (kg * m^2)
        simobj.mass = 133 # (kg)
        simobj.cg = 1.42 # (m)
        x0 = [0, 0, 10000]
        v0 = [1500, 0, 0]
        w0 = [0, 0, 0]
        quat0 = quaternion.from_euler_angles([0, 0, 0]).components
        mass0 = 133.0
        Ixx0 = 1.309
        Iyy0 = 58.27
        Izz0 = 58.27
        gacc0 = -9.81
        simobj.init_state([x0,
                           v0,
                           w0,
                           quat0,
                           mass0,
                           Ixx0, Iyy0, Izz0,
                           gacc0,
                           ])
        self.dt = 0.01
        self.t_span = [0, 0.1]
        return simobj


    def dynamics(self, t, X, U, dt, simobj):
        pos = X[:3]
        vel = X[3:6]
        angvel = X[6:9]
        quat = X[9:13]
        mass = X[13]
        Ixx = X[14]
        Iyy = X[15]
        Izz = X[16]
        gacc = X[17]

        force = U[:3]
        torque = U[3:6]

        acc = force / mass
        angacc = np.array([torque[0]/Ixx,
                           torque[1]/Iyy,
                           torque[2]/Izz])

        gravity = np.array([0, 0, gacc])

        wx, wy, wz = angvel
        Sw = np.array([
            [ 0,   wx,  wy,  wz], #type:ignore
            [-wx,  0,  -wz,  wy], #type:ignore
            [-wy,  wz,   0, -wx], #type:ignore
            [-wz, -wy,  wx,   0], #type:ignore
            ])

        pos_dot = vel
        vel_dot = acc + gravity
        angvel_dot = angacc
        quat_dot = -(0.5 * Sw @ quat)
        mass_dot = 0
        Ixx_dot = 0
        Iyy_dot = 0
        Izz_dot = 0
        gacc_dot = 0

        Xdot = np.array([
            *pos_dot,
            *vel_dot,
            *angvel_dot,
            *quat_dot,
            mass_dot,
            Ixx_dot,
            Iyy_dot,
            Izz_dot,
            gacc_dot,
            ])

        return Xdot


    def run_dynamics(self):
        Nt = int(self.t_span[1] / self.dt)
        t_array = np.linspace(self.t_span[0], self.t_span[1], Nt + 1)
        simobj = self.create_simobj()
        simobj.Y = np.zeros((Nt + 1, len(simobj.X0)))
        simobj.Y[0] = simobj.X0

        U = np.zeros(len(simobj.model.input_vars))

        for istep in range(1, Nt + 1):
            tstep = t_array[istep]
            X = simobj.Y[istep - 1]
            X_new, T_new = runge_kutta_4(
                    f=self.dynamics,
                    t=tstep,
                    X=X,
                    h=self.dt,
                    args=(U, self.dt, simobj,)
                    )
            simobj.Y[istep] = X_new
        truth = simobj.Y
        return truth


if __name__ == '__main__':
    unittest.main()
