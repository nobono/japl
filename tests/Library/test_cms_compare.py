import unittest
import os
import numpy as np
from japl import AeroTable
DIR = os.path.dirname(__file__)



def cosd(deg):
    return np.cos(np.radians(deg))


def sind(deg):
    return np.sin(np.radians(deg))


class TestCmsCompare(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        pass


    def test_invert_aerodynamics_degrees(self):
        """Testing of inverting aerodynamics.
        Static values pulled from CMS unit test."""
        units = ""

        path = f"{DIR}/../../aeromodel/cms_sr_stage1aero.mat"
        aerotable = AeroTable(path,
                              from_template="CMS",
                              units=units)

        alpha = 0.021285852300486
        alpha_max = 80
        alt = 0.097541062161326
        am_c = 4.848711297583447
        mass = 3.040822554083198e+02
        mach = 0.020890665777000
        q_bar = 30.953815676156566
        thrust = 50_000.0
        Sref = 0.05
        # cg = np.nan

        # CN = 0.236450041229858
        # CA = 0.400000000000000
        # CL = 0.224556527015915
        # CD = 0.406796003141811
        # CN_alpha = 0.140346623943120
        # CA_alpha = 0.0
        # CL_alpha = 0.133185708253379
        # CD_alpha = 0.008056236784475

        alpha_tol = .01
        alpha_last = -1000  # last alpha
        count = 0  # iteration counter

        # Gradient search - fixed number of steps
        while ((abs(alpha - alpha_last) > alpha_tol) and (count < 10)):
            count = count + 1  # Update iteration counter
            alpha_last = alpha  # Update last alpha

            # get coeffs from aerotable
            CA = aerotable.get_CA_Boost(alpha=alpha, mach=mach, alt=alt)
            CN = aerotable.get_CNB(alpha=alpha, mach=mach)
            CA_alpha = aerotable.get_CA_Boost_alpha(alpha=alpha, mach=mach, alt=alt)
            CN_alpha = aerotable.get_CNB_alpha(alpha=alpha, mach=mach)

            # Get derivative of cn wrt alpha
            cosa = cosd(alpha)
            sina = sind(alpha)
            CL = (CN * cosa) - (CA * sina)
            CL_alpha = ((CN_alpha - np.radians(CA)) * cosa) - ((CA_alpha + np.radians(CN)) * sina)
            # CD = (CN * sina) + (CA * cosa)  # noqa
            # CD_alpha = ((CA_alpha + np.radians(CN)) * cosa) + ((CN_alpha - np.radians(CA)) * sina)  # noqa

            # Get derivative of missile acceleration normal to flight path wrt alpha
            # d_alpha = alpha_0 - alphaTotal

            am_alpha = CL_alpha * q_bar * Sref / mass + thrust * cosd(alpha) * np.pi / 180 / mass
            am_0 = CL * q_bar * Sref / mass + thrust * sind(alpha) / mass
            alpha = alpha + (am_c - am_0) / am_alpha  # Update angle of attack
            alpha = max(0, min(alpha, alpha_max))  # Bound angle of attack

        # Set output
        aoa = alpha
        out = 1.689392510892652854
        # print()
        # print("-" * 50)
        # print(f"aoa: {aoa}")
        # print(f"tru: {out}")
        # print(f"count: {count}")
        # print(f"alpha error: {abs(alpha - alpha_last)}")
        # print(f"diff: {out - aoa}")
        # print("-" * 50)
        # print()
        self.assertAlmostEqual(aoa, out, self.TOLERANCE_PLACES)


    def test_invert_aerodynamics_radians(self):
        """Testing of inverting aerodynamics.
        Static values pulled from CMS unit test."""
        units = "si"
        path = f"{DIR}/../../aeromodel/cms_sr_stage1aero.mat"
        aerotable = AeroTable(path,
                              from_template="CMS",
                              units=units)

        alpha = np.radians(0.021285852300486)
        alpha_max = np.radians(80)
        alt = 0.097541062161326
        am_c = 4.848711297583447
        mass = 3.040822554083198e+02
        mach = 0.020890665777000
        q_bar = 30.953815676156566
        thrust = 50_000.0
        Sref = 0.05
        # cg = np.nan

        alpha_tol = np.radians(.01)
        alpha_last = -1000  # Last alpha
        count = 0  # Interation counter

        # Gradient search - fixed number of steps
        while ((abs(alpha - alpha_last) > alpha_tol) and (count < 10)):
            count = count + 1  # Update iteration counter
            alpha_last = alpha  # Update last alpha

            # get coeffs from aerotable
            CA = aerotable.get_CA_Boost(alpha=alpha, mach=mach, alt=alt)
            CN = aerotable.get_CNB(alpha=alpha, mach=mach)
            CA_alpha = aerotable.get_CA_Boost_alpha(alpha=alpha, mach=mach, alt=alt)
            CN_alpha = aerotable.get_CNB_alpha(alpha=alpha, mach=mach)

            # Get derivative of cn wrt alpha
            cosa = np.cos(alpha)
            sina = np.sin(alpha)
            CL = (CN * cosa) - (CA * sina)
            CL_alpha = ((CN_alpha - CA) * cosa) - ((CA_alpha + CN) * sina)
            # CD = (CN * sina) + (CA * cosa)  # noqa
            # CD_alpha = ((CA_alpha + CN) * cosa) + ((CN_alpha - CA) * sina)  # noqa

            # Get derivative of missile acceleration normal to flight path wrt alpha
            # d_alpha = alpha_0 - alphaTotal

            am_alpha = CL_alpha * q_bar * Sref / mass + thrust * np.cos(alpha) / mass
            am_0 = CL * q_bar * Sref / mass + thrust * np.sin(alpha) / mass
            alpha = alpha + (am_c - am_0) / am_alpha  # Update angle of attack
            alpha = max(0, min(alpha, alpha_max))  # Bound angle of attack

        # Set output
        aoa = alpha
        out = np.radians(1.689392510892652854)
        # print()
        # print("-" * 50)
        # print(f"aoa: {aoa}")
        # print(f"tru: {out}")
        # print(f"count: {count}")
        # print(f"alpha error: {abs(alpha - alpha_last)}")
        # print(f"diff: {out - aoa}")
        # print("-" * 50)
        # print()
        self.assertAlmostEqual(aoa, out, places=self.TOLERANCE_PLACES)


    def test_invert_aerodynamics(self):
        """Testing of inverting aerodynamics.
        Static values pulled from CMS unit test."""
        units = "si"
        path = f"{DIR}/../../aeromodel/cms_sr_stage1aero.mat"
        aerotable = AeroTable(path,
                              from_template="CMS",
                              units=units)

        alpha = np.radians(0.021285852300486)
        alt = 0.097541062161326
        am_c = 4.848711297583447
        mass = 3.040822554083198e+02
        mach = 0.020890665777000
        q_bar = 30.953815676156566
        thrust = 50_000.0

        angle_of_attack = aerotable.inv_aerodynamics(thrust=thrust,
                                                     acc_cmd=am_c,
                                                     dynamic_pressure=q_bar,
                                                     mass=mass,
                                                     alpha=alpha,
                                                     beta=0,
                                                     phi=0,
                                                     mach=mach,
                                                     alt=alt)
        out = np.radians(1.689392510892652854)
        # print()
        # print("-" * 50)
        # print(f"aoa: {angle_of_attack}")
        # print(f"tru: {out}")
        # print(f"diff: {out - angle_of_attack}")
        # print("-" * 50)
        # print()
        self.assertAlmostEqual(angle_of_attack, out, places=self.TOLERANCE_PLACES)


if __name__ == '__main__':
    unittest.main()
