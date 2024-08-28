import os
import unittest
import numpy as np
from sympy import symbols
from japl.Aero.AeroTableSymbolic import AeroTableSymbolic
from japl.Aero.AeroTable import AeroTable
from japl.Util.Desym import Desym



class TestAeroTableSymbolic(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        aero_file = f"{os.path.dirname(__file__)}/../../aeromodel/aeromodel_psb.mat"
        self.aerotable_sym = AeroTableSymbolic(aero_file)
        self.aerotable = AeroTable(aero_file)
        self.alts = np.linspace(0, 30_000, 100)


    def test_get_CA_Boost_Total(self):
        alpha = np.radians(1)
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach alt iota")
        expr = self.aerotable_sym.get_CA_Boost_Total(*vars)
        get_CA_Boost_Total_sym_func = Desym(vars, expr,  # type:ignore
                                            modules=self.aerotable_sym.modules)
        for alt in self.alts:
            ret1 = self.aerotable.get_CA_Boost_Total(alpha, phi, mach, alt, iota)
            ret2 = get_CA_Boost_Total_sym_func(alpha, phi, mach, alt, iota)
            self.assertEqual(ret1, ret2)


    def test_get_CA_Coast_Total(self):
        alpha = np.radians(1)
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach alt iota")
        expr = self.aerotable_sym.get_CA_Coast_Total(*vars)
        sym_func = Desym(vars, expr,  # type:ignore
                         modules=self.aerotable_sym.modules)
        for alt in self.alts:
            ret1 = self.aerotable.get_CA_Coast_Total(alpha, phi, mach, alt, iota)
            ret2 = sym_func(alpha, phi, mach, alt, iota)
            self.assertEqual(ret1, ret2)


    def test_get_CNB_Total(self):
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach iota")
        expr = self.aerotable_sym.get_CNB_Total(*vars)
        sym_func = Desym(vars, expr,  # type:ignore
                         modules=self.aerotable_sym.modules)
        for alpha in np.linspace(0, .2, 20):
            ret1 = self.aerotable.get_CNB_Total(alpha, phi, mach, iota)
            ret2 = sym_func(alpha, phi, mach, iota)
            self.assertEqual(ret1, ret2)


    def test_get_CLMB_Total(self):
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach iota")
        expr = self.aerotable_sym.get_CLMB_Total(*vars)
        sym_func = Desym(vars, expr,  # type:ignore
                         modules=self.aerotable_sym.modules)
        for alpha in np.linspace(0, .2, 20):
            ret1 = self.aerotable.get_CLMB_Total(alpha, phi, mach, iota)
            ret2 = sym_func(alpha, phi, mach, iota)
            self.assertEqual(ret1, ret2)


    def test_get_CLNB_Total(self):
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach iota")
        expr = self.aerotable_sym.get_CLNB_Total(*vars)
        sym_func = Desym(vars, expr,  # type:ignore
                         modules=self.aerotable_sym.modules)
        for alpha in np.linspace(0, .2, 20):
            ret1 = self.aerotable.get_CLNB_Total(alpha, phi, mach, iota)
            ret2 = sym_func(alpha, phi, mach, iota)
            self.assertEqual(ret1, ret2)


    def test_CNB_Total_alpha(self):
        """Test for CNB table diff wrt. alpha."""
        truth = [32.047116446069630058,
                 31.967586816202516076,
                 29.262716338076089073,
                 25.834808318187693743,
                 23.629769171070027056,
                 19.531822052470484863,
                 15.055218390201394740,
                 1.792828966178075856,
                 0.769799219557401493,
                 12.251060124422552988,
                 15.055218390201394740,
                 19.531822052470609208,
                 23.629769171070027056,
                 25.834808318187693743,
                 29.262716338076089073,
                 31.967586816202516076,
                 32.047116446069630058]

        for i, j in zip(self.aerotable.CNB_Total_alpha[:, 0, 0, 0], truth):
            self.assertAlmostEqual(i, j, places=self.TOLERANCE_PLACES)


if __name__ == '__main__':
    unittest.main()
