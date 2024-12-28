import unittest
import numpy as np
import atmosphere



class TestCppAtmosphere(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15


    def test_tables(self):
        atmos = atmosphere.Atmosphere()
        self.assertAlmostEqual(atmos.pressure(0), 101325.0, places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(atmos.speed_of_sound(1000), 336.43458210225776, places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(atmos.density(1000), 1.1116596736996904, places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(atmos.temperature(1000), 8.501022371694717, places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(atmos.grav_accel(1000), 9.803565306802405, places=self.TOLERANCE_PLACES)


if __name__ == '__main__':
    unittest.main()
