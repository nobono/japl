import unittest
import numpy as np
from sympy import symbols
from japl import SimObject
from japl import Model



class TestSimObject(unittest.TestCase):


    def setUp(self) -> None:
        pass


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
        simobj._set_sim_step(0)
        self.assertEqual(simobj.get_current('x'), 1)
        simobj._set_sim_step(1)
        self.assertEqual(simobj.get_current('y'), 3)
        simobj._set_sim_step(2)
        self.assertEqual(simobj.get_current('z'), 7)



if __name__ == '__main__':
    unittest.main()
