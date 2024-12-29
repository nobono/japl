import os
import unittest
import numpy as np
from pathlib import Path
from japl.Util.Matlab import MatFile
from japl.global_opts import get_root_dir
from japl.AeroTable.AeroTable import AeroTable
from astropy import units as u



class TestAeroTable(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 10
        self.ROOT_DIR = get_root_dir()
        aero_file_path = f"{self.ROOT_DIR}/aerodata/aeromodel_psb.mat"

        # self.aerotable = AeroTable(aero_file_path)
        self.aerotable = AeroTable(aero_file_path,
                                   angle_units=u.deg,  # type:ignore
                                   length_units=u.imperial.foot,  # type:ignore
                                   lref_units=u.imperial.inch)  # type:ignore
        self.alts = np.linspace(0, 30_000, 100)


    def test_case1(self):
        aero = AeroTable()
        self.assertEqual(aero.stages, [])
        self.assertEqual(aero.stage_id, 0)


    def test_stages(self):
        aero = AeroTable()
        stage1 = AeroTable()
        stage2 = AeroTable()
        aero.add_stage(stage1)
        aero.add_stage(stage2)
        self.assertEqual(aero.is_stage, False)
        self.assertEqual(len(aero.stages), 2)
        self.assertEqual(aero.stages[0].is_stage, True)
        self.assertEqual(aero.stages[1].is_stage, True)


    def test_file_type_1(self):
        aero_file_path = Path(self.ROOT_DIR, "aerodata/cms_sr_stage1aero.mat")
        aero = AeroTable(aero_file_path)
        self.assertTrue(not aero.CA_Boost.isnone())
        self.assertTrue(not aero.CA_Coast.isnone())
        self.assertTrue(not aero.CNB.isnone())
        self.assertTrue(not aero.CA_Boost_alpha.isnone())
        self.assertTrue(not aero.CA_Coast_alpha.isnone())
        self.assertTrue(not aero.CNB_alpha.isnone())
        self.assertTrue(hasattr(aero, "Sref"))
        self.assertTrue(hasattr(aero, "Lref"))
        self.assertTrue(hasattr(aero, "MRC"))


    def test_file_type_2(self):
        aero_file_path = Path(self.ROOT_DIR, "aerodata/aeromodel_psb.mat")
        aero = AeroTable(aero_file_path)
        self.assertTrue(not aero.CA_Boost.isnone())
        self.assertTrue(not aero.CA_Coast.isnone())
        self.assertTrue(not aero.CNB.isnone())
        self.assertTrue(not aero.CYB.isnone())
        self.assertTrue(not aero.CLMB.isnone())
        self.assertTrue(not aero.CLNB.isnone())
        self.assertTrue(not aero.CA_Boost_alpha.isnone())
        self.assertTrue(not aero.CA_Coast_alpha.isnone())
        self.assertTrue(not aero.CNB_alpha.isnone())
        self.assertTrue(hasattr(aero, "Sref"))
        self.assertTrue(hasattr(aero, "Lref"))
        self.assertTrue(hasattr(aero, "MRC"))


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
        # aerotable = AeroTable(f"{self.ROOT_DIR}/aerodata/cms_sr_stage1aero.mat",
        #                       from_template="CMS",
        #                       units="english")

        aero_file_path = f"{self.ROOT_DIR}/aerodata/cms_sr_stage1aero.mat"
        aerotable = AeroTable(aero_file_path)
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
