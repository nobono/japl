import unittest
import numpy as np
import quaternion
from japl import Model
from japl import SimObject
from japl import Sim
from japl import AeroTable
from sympy import MatrixSymbol, Matrix, symbols



class TestExample(unittest.TestCase):


    def __build_model_statespace(self):
        model = Model()

        x, y, z = symbols("x y z")              # must be fixed for AeroModel
        vx, vy, vz = symbols("vx vy vz")        # must be fixed for AeroModel
        ax, ay, az = symbols("ax ay az")
        tqx, tqy, tqz = symbols("tqx tqy tqz")
        wx, wy, wz = symbols("wx wy wz")
        q0, q1, q2, q3 = symbols("q0 q1 q2 q3") # must be fixed for AeroModel
        mass = symbols("mass")
        gravx, gravy,  gravz = symbols("gravity_x gravity_y gravity_z")
        dt = symbols("dt")

        Sq = np.array([
            [-q1, -q2, -q3],
            [q0, -q3, q2],
            [q3, q0, -q1],
            [-q2, q1, q0],
            ]) * 0.5

        A = np.array([
            [0,0,0,  1,0,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # x
            [0,0,0,  0,1,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # y
            [0,0,0,  0,0,1,  0,0,0,  0,0,0,0,  0,  0,0,0], # z
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  1,0,0], # vx
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,1,0], # vy
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,0,1], # vz

            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # wx
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # wy
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # wz

            [0,0,0,  0,0,0,  *Sq[0], 0,0,0,0,  0,  0,0,0], # q0
            [0,0,0,  0,0,0,  *Sq[1], 0,0,0,0,  0,  0,0,0], # q1
            [0,0,0,  0,0,0,  *Sq[2], 0,0,0,0,  0,  0,0,0], # q2
            [0,0,0,  0,0,0,  *Sq[3], 0,0,0,0,  0,  0,0,0], # q3

            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # mass

            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # gravityx
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # gravityy
            [0,0,0,  0,0,0,  0,0,0,  0,0,0,0,  0,  0,0,0], # gravityz
            ])

        B = np.array([
            # acc    torque
            [0,0,0,  0,0,0],
            [0,0,0,  0,0,0],
            [0,0,0,  0,0,0],

            [1,0,0,  0,0,0],
            [0,1,0,  0,0,0],
            [0,0,1,  0,0,0],

            [0,0,0,  1,0,0],
            [0,0,0,  0,1,0],
            [0,0,0,  0,0,1],

            [0,0,0,  0,0,0],
            [0,0,0,  0,0,0],
            [0,0,0,  0,0,0],
            [0,0,0,  0,0,0],

            [0,0,0,  0,0,0],

            [0,0,0,  0,0,0],
            [0,0,0,  0,0,0],
            [0,0,0,  0,0,0],
            ])

        state = Matrix([x, y, z, vx, vy, vz, wx, wy, wz, q0, q1, q2, q3, mass,
                        gravx, gravy, gravz])
        input = Matrix([ax, ay, az, tqx, tqy, tqz])
        model.from_statespace(dt, state, input, A, B)
        model.set_state(state)

        vehicle = SimObject(model=model, size=2, color='tab:blue')
        vehicle.aerotable = AeroTable("./aeromodel/aeromodel_psb.mat")

        vehicle.plot.set_config({
                    "Pos": {
                        "xaxis": "x",
                        "yaxis": "z",
                        "aspect": "auto",
                        },
                    "Vel": {
                        "xaxis": "t",
                        "yaxis": "vz",
                        "aspect": "auto",
                        },
                    })

        vehicle.Ixx = 1.309 # (kg * m^2)
        vehicle.Iyy = 58.27 # (kg * m^2)
        vehicle.Izz = 58.27 # (kg * m^2)
        vehicle.mass = 133 # (kg)
        vehicle.cg = 1.42 # (m)
        x0 = [0, 0, 10000]
        v0 = [1500, 0, .5]
        w0 = [0, 0, 0]
        quat0 = quaternion.from_euler_angles([0, 0, 0]).components
        mass0 = 133.0
        gravity0 = [0, 0, -9.81]
        vehicle.init_state([x0, v0, w0, quat0, mass0, gravity0]) # TODO this should be moved to Model

        return vehicle
        

    def __build_model_symbolic(self):
        pos = Matrix(symbols("x y z"))      # must be fixed for AeroModel
        vel = Matrix(symbols("vx vy vz"))   # must be fixed for AeroModel
        acc = Matrix(symbols("ax ay az"))
        tq = Matrix(symbols("tqx tqy tqz"))
        w = Matrix(symbols("wx wy wz"))
        q = Matrix(symbols("q0 q1 q2 q3"))  # must be fixed for AeroModel
        gravity = Matrix(symbols("gravity_x gravity_y gravity_z"))

        dt = symbols("dt")
        mass = symbols("mass")

        w_skew = Matrix(w).hat()        #type:ignore
        Sw = Matrix(np.zeros((4,4)))
        Sw[0, :] = Matrix([0, *w]).T
        Sw[:, 0] = Matrix([0, *-w])     #type:ignore
        Sw[1:, 1:] = w_skew

        x_new = pos + vel * dt
        v_new = vel + (acc + gravity) * dt
        w_new = w + tq * dt
        q_new = q + (-0.5 * Sw * q) * dt

        X_new = Matrix([
            x_new.as_mutable(),
            v_new.as_mutable(),
            w_new.as_mutable(),
            q_new.as_mutable(),
            mass,
            gravity,
            ])

        state = Matrix([pos, vel, w, q, mass, gravity])
        input = Matrix([acc, tq])

        dynamics = X_new.diff(dt)

        model = Model().from_expression(dt, state, input, dynamics)

        vehicle = SimObject(model=model, size=2, color='tab:blue')
        vehicle.aerotable = AeroTable("./aeromodel/aeromodel_psb.mat")

        vehicle.plot.set_config({
                    "Pos": {
                        "xaxis": "x",
                        "yaxis": "z",
                        "aspect": "auto",
                        },
                    "Vel": {
                        "xaxis": "t",
                        "yaxis": "vz",
                        "aspect": "auto",
                        },
                    })

        vehicle.Ixx = 1.309 # (kg * m^2)
        vehicle.Iyy = 58.27 # (kg * m^2)
        vehicle.Izz = 58.27 # (kg * m^2)
        vehicle.mass = 133 # (kg)
        vehicle.cg = 1.42 # (m)
        x0 = [0, 0, 10000]
        v0 = [1500, 0, .5]
        w0 = [0, 0, 0]
        quat0 = quaternion.from_euler_angles([0, 0, 0]).components
        mass0 = 133.0
        gravity0 = [0, 0, -9.81]
        vehicle.init_state([x0, v0, w0, quat0, mass0, gravity0]) # TODO this should be moved to Model

        return vehicle

    def run_example(self, vehicle: SimObject):
        sim = Sim(
                t_span=[0, 0.1],
                dt=.01,
                simobjs=[vehicle],
                integrate_method="rk4",
                events=[],
                animate=0,
                aspect="equal",
                device_input_type="",
                moving_bounds=True,
                rtol=1e-6,
                atol=1e-6,
                blit=False,
                antialias=0,
                figsize=(10, 7),
                instrument_view=1,
                draw_cache_mode=0,
                quiet=1, # TODO still working on this
                )
        sim.run()
        return sim


    def test_compare(self):
        vehicle_ss = self.__build_model_statespace()
        vehicle_sym = self.__build_model_symbolic()
        sim_ss = self.run_example(vehicle_ss)
        sim_sym = self.run_example(vehicle_sym)

        truth = np.array([
            150.00000000000014210855,
            0.00000000000000000000,
            10000.00111889933941711206,
            1500.00000000000000000000,
            0.00000000000000000000,
            -0.47741315793600341832,
            0.00000000000000000000,
            0.00020871625291801776,
            0.00000000000000000000,
            0.99999999999756350455,
            0.00000000000000000000,
            0.00000221291092111800,
            0.00000000000000000000,
            133.00000000000000000000,
            0.00000000000000000000,
            0.00000000000000000000,
            -9.77586844288743428422,
            ])

        # symbolic expressions should be equivalent
        self.assertTrue(vehicle_ss.model.expr == vehicle_sym.model.expr)

        # check if state histories match
        for i in range(len(vehicle_ss.Y)):
            comp = vehicle_ss.Y[i] == vehicle_sym.Y[i]
            self.assertTrue(comp.all())

        # check last state entry
        self.assertTrue(np.linalg.norm(vehicle_ss.Y[-1] - truth) < 1e-18)
        self.assertTrue(np.linalg.norm(vehicle_sym.Y[-1] - truth) < 1e-18)


if __name__ == '__main__':
    unittest.main()
