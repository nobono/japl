import unittest
import numpy as np
from sympy import symbols
from sympy import Matrix
from japl import SimObject
from japl import Model
from japl.Util import noprint



class TestSimObject(unittest.TestCase):


    def setUp(self) -> None:
        pass


    @noprint
    def setup_model(self):
        dt, pos, vel, acc = symbols("dt, pos, vel, acc")
        state = Matrix([pos, vel])
        input = Matrix([acc])
        pos_new = pos + vel * dt
        vel_new = vel + acc * dt
        state_new = Matrix([pos_new, vel_new])
        dynamics = state_new.diff(dt)  # type:ignore
        model = Model.from_expression(dt_var=dt,
                                      state_vars=state,
                                      input_vars=input,
                                      dynamics_expr=dynamics,  # type:ignore
                                      use_multiprocess_build=False)

        def pre_update_func(t, X, U, S, dt):
            return []

        def input_func(t, X, U, S, dt):
            return []

        def post_update_func(t, X, U, S, dt):
            return []

        model.pre_update_functions += [pre_update_func]
        model.input_function = input_func
        model.post_update_functions += [post_update_func]

        # cache functions
        model.cache_build(use_parallel=False)
        return model


    def test_instantiate_case1(self):
        simobj = SimObject()
        self.assertTrue(simobj.color)


    def test_instantiate_case2(self):
        simobj = SimObject(name="test_obj", color="blue", size=2)
        self.assertTrue(simobj.name, "test_obj")
        self.assertTrue(simobj.color, "blue")
        self.assertTrue(simobj.size, 2)


    def test_instantiate_set_draw(self):
        simobj = SimObject()
        simobj.set_draw(color="blue", size=2)
        self.assertTrue(simobj.color, "blue")
        self.assertTrue(simobj.size, 2)


    def test_get(self):
        x, y, z = symbols("x y z")
        simobj = SimObject()
        simobj.model.set_state([x, y, z])
        simobj.X0 = np.array([0, 1, 2])
        simobj.U0 = np.array([3, 4])
        simobj.Y = np.array([[1, 2, 5],
                             [2, 3, 6],
                             [3, 4, 7]])
        self.assertListEqual(simobj.get('x').tolist(), [1, 2, 3])
        self.assertListEqual(simobj.get('y').tolist(), [2, 3, 4])
        self.assertListEqual(simobj.get('z').tolist(), [5, 6, 7])


    def test_get_case2(self):
        x, y, z = symbols("x y z")
        simobj = SimObject()
        simobj.model.set_state([x, y, z])
        simobj.X0 = np.array([0, 1, 2])
        simobj.U0 = np.array([3, 4])
        simobj.Y = np.array([[1, 2, 5],
                             [2, 3, 6],
                             [3, 4, 7]])
        self.assertListEqual(simobj.get('x, y').tolist(), [[1, 2],
                                                           [2, 3],
                                                           [3, 4]])
        self.assertListEqual(simobj.get('x z').tolist(), [[1, 5],
                                                          [2, 6],
                                                          [3, 7]])
        self.assertListEqual(simobj.get('x,,  y,').tolist(), [[1, 2],
                                                              [2, 3],
                                                              [3, 4]])


    def test_current(self):
        x, y, z = symbols("x y z")
        simobj = SimObject()
        simobj.model.set_state([x, y, z])
        simobj.X0 = np.array([0, 1, 2])
        simobj.U0 = np.array([3, 4])
        simobj.Y = np.array([[1, 2, 5],
                             [2, 3, 6],
                             [3, 4, 7]])
        simobj.set_istep(0)
        self.assertEqual(simobj.get_current('x'), 1)
        simobj.set_istep(1)
        self.assertEqual(simobj.get_current('y'), 3)
        simobj.set_istep(2)
        self.assertEqual(simobj.get_current('z'), 7)


    # def test_get_sim_func_call_list(self):
    #     model = self.setup_model()
    #     simobj = SimObject(model)
    #     call_list = simobj._get_sim_func_call_list()
    #     self.assertEqual(len(call_list), 6)


    # def test_connect_models(self):
    #     s1 = SimObject(self.setup_model())
    #     s2 = SimObject(self.setup_model())
    #     s3 = SimObject(self.setup_model())
    #     # connect models --------------------------------------------------
    #     s2.children_pre_update = [s3]
    #     s1.children_pre_update = [s2]
    #     call_list = s1._get_sim_func_call_list()
    #     self.assertEqual(len(call_list), 18)
    #     self.assertEqual(call_list[0].__name__, "pre_update_func")
    #     self.assertEqual(call_list[1].__name__, "input_func")
    #     self.assertEqual(call_list[2].__name__, "input_updates")
    #     self.assertEqual(call_list[3].__name__, "state_updates")
    #     self.assertEqual(call_list[4].__name__, "dynamics")
    #     self.assertEqual(call_list[5].__name__, "post_update_func")
    #     # model 2 ---------------------------------------------------------
    #     self.assertEqual(call_list[6].__name__, "pre_update_func")
    #     self.assertEqual(call_list[7].__name__, "input_func")
    #     self.assertEqual(call_list[8].__name__, "input_updates")
    #     self.assertEqual(call_list[9].__name__, "state_updates")
    #     self.assertEqual(call_list[10].__name__, "dynamics")
    #     self.assertEqual(call_list[11].__name__, "post_update_func")
    #     # model 3 ---------------------------------------------------------
    #     self.assertEqual(call_list[12].__name__, "pre_update_func")
    #     self.assertEqual(call_list[13].__name__, "input_func")
    #     self.assertEqual(call_list[14].__name__, "input_updates")
    #     self.assertEqual(call_list[15].__name__, "state_updates")
    #     self.assertEqual(call_list[16].__name__, "dynamics")
    #     self.assertEqual(call_list[17].__name__, "post_update_func")
    #     # ensure model functions are unique -------------------------------
    #     self.assertNotEqual(id(call_list[0]), id(call_list[6]))
    #     self.assertNotEqual(id(call_list[1]), id(call_list[7]))
    #     self.assertNotEqual(id(call_list[2]), id(call_list[8]))
    #     self.assertNotEqual(id(call_list[3]), id(call_list[9]))
    #     self.assertNotEqual(id(call_list[4]), id(call_list[10]))
    #     self.assertNotEqual(id(call_list[5]), id(call_list[11]))
    #     self.assertNotEqual(id(call_list[12]), id(call_list[6]))
    #     self.assertNotEqual(id(call_list[13]), id(call_list[7]))
    #     self.assertNotEqual(id(call_list[14]), id(call_list[8]))
    #     self.assertNotEqual(id(call_list[15]), id(call_list[9]))
    #     self.assertNotEqual(id(call_list[16]), id(call_list[10]))
    #     self.assertNotEqual(id(call_list[17]), id(call_list[11]))


if __name__ == '__main__':
    unittest.main()
