import unittest
from japl.Library.Earth.EarthModelSymbolic import EarthModelSymbolic



class TestEarthModel(unittest.TestCase):


    def setUp(self) -> None:
        self.earth = EarthModelSymbolic()


    def test_instantiate(self):
        E = self.earth
        self.assertEqual(E.mu, 3986004.418e8)
        self.assertEqual(E.J2, 0.10826299890519e-2)
        self.assertEqual(E.omega, 7.2921159e-5)
        self.assertEqual(E.semimajor_axis, 6_378_137.0)
        self.assertEqual(E.inv_flattening, 298.257223563)
        self.assertEqual(E.flattening, 1 / 298.257223563)


if __name__ == '__main__':
    unittest.main()
