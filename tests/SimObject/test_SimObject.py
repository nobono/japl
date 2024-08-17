import unittest
from japl import SimObject



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


if __name__ == '__main__':
    unittest.main()
