import os
import unittest
import numpy as np
from japl.Aero.AeroTable import AeroTable



class TestAeroTable(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 10
        self.DIR = os.path.dirname(__file__)
        aero_file = f"{self.DIR}/../../aeromodel/aeromodel_psb.mat"
        self.aerotable = AeroTable(aero_file)
        self.alts = np.linspace(0, 30_000, 100)


    def test_CNB_alpha(self):
        """Test for CNB table diff wrt. alpha."""
        truth = [-32.047116446069630058,
                 -31.967586816202516076,
                 -29.262716338076089073,
                 -25.834808318187693743,
                 -23.629769171070027056,
                 -19.531822052470609208,
                 -15.055218390201394740,
                 -12.251060124422552988,
                 11.228030377801893280,
                 12.251060124422552988,
                 15.055218390201394740,
                 19.531822052470609208,
                 23.629769171070027056,
                 25.834808318187693743,
                 29.262716338076089073,
                 31.967586816202516076,
                 32.047116446069630058]

        for i, j in zip(self.aerotable.CNB_alpha[:, 0, 0, 0], truth):
            self.assertAlmostEqual(i, j, places=self.TOLERANCE_PLACES)


    def test_CMS_getAoA_compare(self):
        """This test, tests against CMS's getAoA method."""
        aerotable = AeroTable(f"{self.DIR}/../../aeromodel/cms_sr_stage1aero.mat",
                              from_template="CMS",
                              units="english")
        CN = 0.236450041229858
        CA = 0.400000000000000
        CN_alpha = 0.140346623943120

        alpha = 1.689147711404596
        mach = 0.020890665777000
        alt = 0.097541062161326

        self.assertTrue((aerotable.CA_Coast_alpha == 0).all())
        self.assertTrue((aerotable.CA_Boost_alpha == 0).all())
        self.assertAlmostEqual(aerotable.CA_Boost(alpha=alpha, mach=mach, alt=alt),
                               CA,
                               places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(aerotable.CNB(alpha=alpha, mach=mach, alt=alt),
                               CN,
                               places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(aerotable.CNB_alpha(alpha=alpha, mach=mach, alt=alt),
                               CN_alpha,
                               places=self.TOLERANCE_PLACES)


if __name__ == '__main__':
    unittest.main()
