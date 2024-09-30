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
        self.aerotable_sym = AeroTableSymbolic()
        self.aerotable = AeroTable(aero_file)
        self.alts = np.linspace(0, 30_000, 100)


    def test_get_CA_Boost(self):
        alpha = np.radians(1)
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach alt iota")
        expr = self.aerotable_sym.get_CA_Boost(*vars)
        get_CA_Boost_sym_func = Desym(vars, expr,  # type:ignore
                                      modules=self.aerotable.modules)
        for alt in self.alts:
            ret1 = self.aerotable.get_CA_Boost(alpha, phi, mach, alt, iota)
            ret2 = get_CA_Boost_sym_func(alpha, phi, mach, alt, iota)
            self.assertEqual(ret1, ret2)


    def test_get_CA_Coast(self):
        alpha = np.radians(1)
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach alt iota")
        expr = self.aerotable_sym.get_CA_Coast(*vars)
        sym_func = Desym(vars, expr,  # type:ignore
                         modules=self.aerotable.modules)
        for alt in self.alts:
            ret1 = self.aerotable.get_CA_Coast(alpha, phi, mach, alt, iota)
            ret2 = sym_func(alpha, phi, mach, alt, iota)
            self.assertEqual(ret1, ret2)


    def test_get_CNB(self):
        phi = 0
        mach = 1.5
        alt = 0
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach alt iota")
        expr = self.aerotable_sym.get_CNB(*vars)
        sym_func = Desym(vars, expr,  # type:ignore
                         modules=self.aerotable.modules)
        for alpha in np.linspace(0, .2, 20):
            ret1 = self.aerotable.get_CNB(alpha, phi, mach, alt, iota)
            ret2 = sym_func(alpha, phi, mach, alt, iota)
            self.assertEqual(ret1, ret2)


    def test_get_CLMB(self):
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach iota")
        expr = self.aerotable_sym.get_CLMB(*vars)
        sym_func = Desym(vars, expr,  # type:ignore
                         modules=self.aerotable.modules)
        for alpha in np.linspace(0, .2, 20):
            ret1 = self.aerotable.get_CLMB(alpha, phi, mach, iota)
            ret2 = sym_func(alpha, phi, mach, iota)
            self.assertEqual(ret1, ret2)


    def test_get_CLNB(self):
        phi = 0
        mach = 1.5
        iota = np.radians(0.1)
        vars = symbols("alpha phi mach iota")
        expr = self.aerotable_sym.get_CLNB(*vars)
        sym_func = Desym(vars, expr,  # type:ignore
                         modules=self.aerotable.modules)
        for alpha in np.linspace(0, .2, 20):
            ret1 = self.aerotable.get_CLNB(alpha, phi, mach, iota)
            ret2 = sym_func(alpha, phi, mach, iota)
            self.assertEqual(ret1, ret2)


if __name__ == '__main__':
    unittest.main()
