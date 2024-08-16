import unittest
from sympy import symbols, Matrix, Function, Symbol
from japl.BuildTools import BuildTools
# from japl.BuildTools.DirectUpdate import DirectUpdate
# from japl.BuildTools.DirectUpdate import DirectUpdateSymbol
# from japl import Model
# from pprint import pprint



class TestBuildTools(unittest.TestCase):


    def setUp(self) -> None:
        pass


    def setup_symbols(self):
        t = symbols("t")
        dt = symbols("dt")
        pos_x = Function("pos_x", real=True)(t)  # type:ignore
        pos_y = Function("pos_y", real=True)(t)  # type:ignore
        pos_z = Function("pos_z", real=True)(t)  # type:ignore
        vel_x = Function("vel_x", real=True)(t)  # type:ignore
        vel_y = Function("vel_y", real=True)(t)  # type:ignore
        vel_z = Function("vel_z", real=True)(t)  # type:ignore
        acc_x = Symbol("acc_x", real=True)  # type:ignore
        acc_y = Symbol("acc_y", real=True)  # type:ignore
        acc_z = Symbol("acc_z", real=True)  # type:ignore
        pos = Matrix([pos_x,
                      pos_y,
                      pos_z])
        vel = Matrix([vel_x,
                      vel_y,
                      vel_z])
        acc = Matrix([acc_x,
                      acc_y,
                      acc_z])
        return (t, dt, pos, vel, acc)


    def run_model(self, state: Matrix, input: Matrix, dynamics: Matrix):
        # model = Model.from_expression(dt, state, input, dynamics,
        #                               definitions=defs)
        # subs = {k: 0 for k, v}
        # model.dynamics_expr.subs(subs)
        pass


    def test_BuildTools_state_array_checks_case1(self):
        (t, dt, pos, vel, acc) = self.setup_symbols()

        pos_new = pos + vel * dt
        vel_new = vel + acc * dt
        pos_dot = pos_new.diff(dt)
        vel_dot = vel_new.diff(dt)
        defs = (
                (pos.diff(t), pos_dot),
                (vel.diff(t), vel_dot),
                )
        state = Matrix([pos, vel])
        input = Matrix([acc])
        dynamics = Matrix(state.diff(t))

        state, input, dynamics = BuildTools.build_model(state,
                                                        input,
                                                        dynamics,
                                                        defs)
        names = BuildTools._get_array_var_names(state)

        # self.run_model(state_vars, input_vars, dynamics_expr)

        # model = Model.from_expression(dt, state, input, dynamics,
        #                               definitions=defs)
        # pprint(model.dynamics_expr)


if __name__ == '__main__':
    unittest.main()
