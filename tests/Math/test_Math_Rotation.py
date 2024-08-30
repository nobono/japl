import unittest
import numpy as np
import quaternion
from japl.Math.Rotation import quat_conj
from japl.Math.Rotation import quat_mult
from japl.Math.Rotation import quat_to_dcm
from japl.Math.Rotation import quat_to_tait_bryan
from japl.Math.Rotation import tait_bryan_to_dcm
from japl.Math.Rotation import dcm_to_tait_bryan
from japl.Math.Rotation import eci_to_ecef
from japl.Math.Rotation import ecef_to_eci
from japl.Math.Rotation import ecef_to_lla
from japl.Math.Rotation import ecef_to_enu
from japl.Math.Rotation import lla_to_ecef
from sympy import Matrix, MatrixSymbol
from japl.Math.RotationSymbolic import quat_conj_sym
from japl.Math.RotationSymbolic import quat_norm_sym
from japl.Math.RotationSymbolic import quat_mult_sym
from japl.Math.RotationSymbolic import quat_to_dcm_sym
from japl.Math.RotationSymbolic import quat_to_tait_bryan_sym
from japl.Math.RotationSymbolic import tait_bryan_to_dcm_sym
from japl.Math.RotationSymbolic import dcm_to_tait_bryan_sym
from japl.Math.RotationSymbolic import ecef_to_lla_sym

np.set_printoptions(suppress=True, precision=18)



class TestMathRotation(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        self.TOLERANCE = 1e-15


    def test_quat_conj(self):
        quat = quaternion.from_euler_angles(0, 0, np.radians(45))
        conj = quat_conj(quat.components)
        self.assertTrue(quat.w == conj[0])
        self.assertTrue((quat.components[1:] == -conj[1:]).all())


    def test_quat_norm(self):
        """compare numpy quaternion method
        with symbolic method."""
        # numpy
        q = np.array([1, 2, 3, 4])
        quat = quaternion.from_float_array(q)
        quat_normalized = quat.normalized().components
        # symbolic
        quat_sym = Matrix(q)
        quat_normalized_sym = quat_norm_sym(quat_sym)
        subs = {quat_sym[0]: q[0],  # type:ignore
                quat_sym[1]: q[1],  # type:ignore
                quat_sym[2]: q[2],  # type:ignore
                quat_sym[3]: q[3]}  # type:ignore
        self.assertListEqual(quat_normalized.tolist(), quat_normalized_sym.subs(subs).n().flat())


    def test_quat_mult(self):
        quat1 = quaternion.from_euler_angles(np.radians(5), np.radians(0), np.radians(10))
        quat2 = quaternion.from_euler_angles(np.radians(5), np.radians(10), np.radians(20))
        ret = quat_mult(quat1.components, quat2.components)
        truth = np.array([
            0.936116806662859124,
            0.022557566113149831,
            0.084185982829369177,
            0.340718653421610063,
            ])
        self.assertAlmostEqual(ret[0], truth[0], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(ret[1], truth[1], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(ret[2], truth[2], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(ret[3], truth[3], places=self.TOLERANCE_PLACES)


    def test_quat_to_dcm(self):
        # quat = quaternion.from_euler_angles(*np.radians([10, 20, 30]))
        quat = quaternion.from_float_array([0.925416578398323364,
                                            0.030153689607045796,
                                            0.171010071662834329,
                                            0.336824088833465152])
        dcm = quat_to_dcm(quat)
        truth = np.array([[0.7146101771427564, -0.6130920223795969, 0.3368240888334651],
                          [0.633718360861996, 0.7712805763691759, 0.0593911746138847],
                          [-0.29619813272602374, 0.17101007166283433, 0.9396926207859084]])
        for i, j in zip(dcm.flatten(), truth.flatten()):
            self.assertAlmostEqual(i, j, places=self.TOLERANCE_PLACES)


    def test_tait_bryan_to_dcm(self):
        yaw_pitch_roll = np.radians([10, 20, 30])
        dcm = tait_bryan_to_dcm(yaw_pitch_roll)
        truth = np.array([[0.925416578398323364, 0.018028311236297251, 0.378522306369792449],
                          [0.163175911166534821, 0.882564119259385604, -0.440969610529882372],
                          [-0.342020143325668713, 0.469846310392954158, 0.813797681349373803]])
        self.assertListEqual(dcm.flatten().tolist(), truth.flatten().tolist())


    def test_dcm_to_tait_bryan(self):
        # quat = quaternion.from_euler_angles(*np.radians([10, 20, 30]))
        quat = quaternion.from_float_array([0.925416578398323364,
                                            0.030153689607045796,
                                            0.171010071662834329,
                                            0.336824088833465152])
        dcm = quaternion.as_rotation_matrix(quat)
        tb_angles = dcm_to_tait_bryan(dcm)
        truth = np.array([0.725475843410945509, 0.300709698155427141, 0.180015088428340159])
        self.assertListEqual(tb_angles.tolist(), truth.tolist())


    def test_quat_to_tait_bryan(self):
        quat = quaternion.from_float_array([0.925416578398323364,
                                            0.030153689607045796,
                                            0.171010071662834329,
                                            0.336824088833465152])
        tb_angles = quat_to_tait_bryan(quat)
        truth = np.array([0.725475843410945509, 0.300709698155427141, 0.180015088428340159])
        self.assertListEqual(tb_angles.tolist(), truth.tolist())


    def __angles_to_dcm_to_quat_to_dcm_to_angles(self, tait_bryan_angles):
        dcm = tait_bryan_to_dcm(tait_bryan_angles)
        q = quaternion.from_rotation_matrix(dcm)
        dcm = quat_to_dcm(q)
        tb_angles = dcm_to_tait_bryan(dcm)
        return tb_angles


    def test_quat_to_tait_bryan_case1(self):
        tait_bryan_angles = np.radians([0, 45, 0])
        tb_angles_out = self.__angles_to_dcm_to_quat_to_dcm_to_angles(tait_bryan_angles)
        self.assertAlmostEqual(tb_angles_out[0], tait_bryan_angles[0], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(tb_angles_out[1], tait_bryan_angles[1], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(tb_angles_out[2], tait_bryan_angles[2], places=self.TOLERANCE_PLACES)

        tait_bryan_angles = np.radians([45, 0, 0])
        tb_angles_out = self.__angles_to_dcm_to_quat_to_dcm_to_angles(tait_bryan_angles)
        self.assertAlmostEqual(tb_angles_out[0], tait_bryan_angles[0], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(tb_angles_out[1], tait_bryan_angles[1], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(tb_angles_out[2], tait_bryan_angles[2], places=self.TOLERANCE_PLACES)

        tait_bryan_angles = np.radians([0, 0, 45])
        tb_angles_out = self.__angles_to_dcm_to_quat_to_dcm_to_angles(tait_bryan_angles)
        self.assertAlmostEqual(tb_angles_out[0], tait_bryan_angles[0], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(tb_angles_out[1], tait_bryan_angles[1], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(tb_angles_out[2], tait_bryan_angles[2], places=self.TOLERANCE_PLACES)


    def test_tait_bryan_angles_case2(self):
        tait_bryan_angles = np.array([10, 90, 5])
        tb_angles_out = self.__angles_to_dcm_to_quat_to_dcm_to_angles(np.radians(tait_bryan_angles))
        tb_angles_out = np.degrees(tb_angles_out)
        self.assertAlmostEqual(tb_angles_out[0], 175.0, places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(tb_angles_out[1], 90.0, places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(tb_angles_out[2], 0.0, places=self.TOLERANCE_PLACES)

    ###################################################################################
    # Symbolic
    ###################################################################################

    def test_quat_conj_sym(self):
        quat = MatrixSymbol('q', 4, 1)
        conj = quat_conj_sym(quat)
        self.assertTrue(quat[0] == conj[0])
        self.assertTrue(quat[1] == -conj[1])  # type:ignore
        self.assertTrue(quat[2] == -conj[2])  # type:ignore
        self.assertTrue(quat[3] == -conj[3])  # type:ignore


    def test_quat_mult_sym(self):
        quat1 = MatrixSymbol('q', 4, 1)
        quat2 = MatrixSymbol('p', 4, 1)
        quat1_ = quaternion.from_euler_angles(np.radians(5), np.radians(0), np.radians(10)).components
        quat2_ = quaternion.from_euler_angles(np.radians(5), np.radians(10), np.radians(20)).components
        qsubs = {quat1[0]: quat1_[0],
                 quat1[1]: quat1_[1],
                 quat1[2]: quat1_[2],
                 quat1[3]: quat1_[3]}
        psubs = {quat2[0]: quat2_[0],
                 quat2[1]: quat2_[1],
                 quat2[2]: quat2_[2],
                 quat2[3]: quat2_[3]}
        ret = quat_mult_sym(quat1, quat2)
        ret = ret.subs(qsubs).subs(psubs)
        ret = np.asarray(ret).squeeze()
        truth = np.array([
            0.936116806662859124,
            0.022557566113149831,
            0.084185982829369177,
            0.340718653421610063,
            ])
        self.assertAlmostEqual(ret[0], truth[0], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(ret[1], truth[1], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(ret[2], truth[2], places=self.TOLERANCE_PLACES)
        self.assertAlmostEqual(ret[3], truth[3], places=self.TOLERANCE_PLACES)


    def test_quat_to_dcm_sym(self):
        quat = MatrixSymbol("q", 4, 1).as_mutable()
        dcm = quat_to_dcm_sym(quat)
        subs = {quat[0]: 0.925416578398323364,  # type:ignore
                quat[1]: 0.030153689607045796,  # type:ignore
                quat[2]: 0.171010071662834329,  # type:ignore
                quat[3]: 0.336824088833465152}  # type:ignore
        truth = np.array([[0.7146101771427564, -0.6130920223795969, 0.3368240888334651],
                          [0.633718360861996, 0.7712805763691759, 0.0593911746138847],
                          [-0.29619813272602374, 0.17101007166283433, 0.9396926207859084]])
        dcm_subbed = np.array(dcm.subs(subs)).squeeze().astype(float)
        for i, j in zip(dcm_subbed.flatten(), truth.flatten()):
            self.assertAlmostEqual(i, j, places=self.TOLERANCE_PLACES)


    def test_tait_bryan_to_dcm_sym(self):
        yaw_pitch_roll = MatrixSymbol("yaw_pitch_roll", 3, 1).as_mutable()
        dcm = tait_bryan_to_dcm_sym(yaw_pitch_roll)
        subs = {yaw_pitch_roll[0]: np.radians(10),  # type:ignore
                yaw_pitch_roll[1]: np.radians(20),  # type:ignore
                yaw_pitch_roll[2]: np.radians(30)}  # type:ignore
        truth = np.array([[0.925416578398323364, 0.018028311236297251, 0.378522306369792449],
                          [0.163175911166534821, 0.882564119259385604, -0.440969610529882372],
                          [-0.342020143325668713, 0.469846310392954158, 0.813797681349373803]])
        dcm_subbed = np.array(dcm.subs(subs)).squeeze().astype(float)
        for i, j in zip(dcm_subbed.flatten(), truth.flatten()):
            self.assertAlmostEqual(i, j, places=self.TOLERANCE_PLACES)


    def test_dcm_to_tait_bryan_sym(self):
        quat = MatrixSymbol("q", 4, 1).as_mutable()
        dcm = quat_to_dcm_sym(quat)
        subs = {quat[0]: 0.925416578398323364,  # type:ignore
                quat[1]: 0.030153689607045796,  # type:ignore
                quat[2]: 0.171010071662834329,  # type:ignore
                quat[3]: 0.336824088833465152}  # type:ignore
        tb_angles = dcm_to_tait_bryan_sym(dcm)
        truth = np.array([0.725475843410945509,
                          0.300709698155427141,
                          0.180015088428340159])
        tb_angles_subbed = np.array(tb_angles.subs(subs)).squeeze().astype(float)
        for i, j in zip(tb_angles_subbed.flatten(), truth.flatten()):
            self.assertAlmostEqual(i, j, places=self.TOLERANCE_PLACES)


    def test_quat_to_tait_bryan_sym(self):
        quat = MatrixSymbol("q", 4, 1).as_mutable()
        subs = {quat[0]: 0.925416578398323364,  # type:ignore
                quat[1]: 0.030153689607045796,  # type:ignore
                quat[2]: 0.171010071662834329,  # type:ignore
                quat[3]: 0.336824088833465152}  # type:ignore
        tb_angles = quat_to_tait_bryan_sym(quat)
        truth = np.array([0.725475843410945509,
                          0.300709698155427141,
                          0.180015088428340159])
        tb_angles_subbed = np.array(tb_angles.subs(subs)).squeeze().astype(float)
        for i, j in zip(tb_angles_subbed.flatten(), truth.flatten()):
            self.assertAlmostEqual(i, j, places=self.TOLERANCE_PLACES)


    def test_eci_to_ecef(self):
        t = 0
        radius_equatorial = 6_378_137.0  # meters
        eci = np.array([radius_equatorial, 0, 0])
        ecef = eci_to_ecef(eci, t=t)
        self.assertListEqual(eci.tolist(), ecef.tolist())


    def test_eci_to_ecef_case2(self):
        t = 1
        radius_equatorial = 6_378_137.0  # meters
        eci = np.array([radius_equatorial, 0, 0])
        ecef = eci_to_ecef(eci, t=t)
        truth_eci = [6378137., 0., 0.]
        truth_ecef = [6378136.983042143, -465.1011418885875, 0.]
        self.assertListEqual(eci.tolist(), truth_eci)
        self.assertListEqual(ecef.tolist(), truth_ecef)


    def test_ecef_to_eci(self):
        t = 0
        radius_equatorial = 6_378_137.0  # meters
        ecef = np.array([radius_equatorial, 0, 0])  # ecef @ t=0
        eci = ecef_to_eci(ecef, t=t)
        self.assertListEqual(eci.tolist(), ecef.tolist())


    def test_ecef_to_eci_case2(self):
        t = 1
        radius_equatorial = 6_378_137.0  # meters
        ecef = np.array([radius_equatorial, 0, 0])  # ecef @ t=0
        eci = ecef_to_eci(ecef, t=t)
        truth_eci = [6378136.983042143, 465.1011418885875, 0.]
        truth_ecef = [6378137., 0., 0.]
        self.assertListEqual(eci.tolist(), truth_eci)
        self.assertListEqual(ecef.tolist(), truth_ecef)


    def test_ecef_to_lla(self):
        radius_equatorial = 6_378_137.0  # meters
        ecef = np.array([radius_equatorial, 0, 0])  # ecef @ t=0
        lla = ecef_to_lla(ecef)
        self.assertListEqual(lla.tolist(), [0, 0, 0])


    def test_ecef_to_lla_case2(self):
        radius_equatorial = 6_378_137.0  # meters
        ecef = np.array([radius_equatorial, 2000, 100])  # ecef @ t=0
        lla = ecef_to_lla(ecef)
        truth = [0.000015784224246508,
                 0.000313571178299987,
                 0.31436039186129777]
        for i, j in zip(lla, truth):
            self.assertAlmostEqual(i, j, places=self.TOLERANCE_PLACES)


    def test_ecef_to_enu(self):
        radius_equatorial = 6_378_137.0  # meters
        ecef0 = np.array([radius_equatorial, 0, 0])  # ecef @ t=0
        ecef = np.array([radius_equatorial + 1, 0, 0])  # ecef @ t=0
        enu = ecef_to_enu(ecef, ecef0)
        east, north, up = enu
        self.assertEqual(east, 0)
        self.assertEqual(north, 0)
        self.assertEqual(up, 1)


    def test_lla_to_ecef(self):
        radius_equatorial = 6_378_137.0  # meters
        lla = np.array([0, 0, 0])
        ecef = lla_to_ecef(lla)
        truth_ecef = np.array([radius_equatorial, 0, 0])  # ecef @ t=0
        self.assertListEqual(ecef.tolist(), truth_ecef.tolist())


    def test_lla_to_ecef_case2(self):
        radius_equatorial = 6_378_137.0  # meters
        lla = np.array([0.000015784224246508,
                        0.000313571178299987,
                        0.31436039186129777])
        ecef = lla_to_ecef(lla)
        truth_ecef = np.array([radius_equatorial, 2000, 100])
        # this method contiains higher inaccuracies
        for i, j in zip(ecef, truth_ecef):
            self.assertAlmostEqual(i, j, places=8)

    #################################
    # Symbolic
    #################################

    def test_ecef_to_lla_sym(self):
        radius_equatorial = 6_378_137.0  # meters
        ecef = Matrix([radius_equatorial, 0, 0])  # ecef @ t=0
        lla = ecef_to_lla_sym(ecef)
        self.assertListEqual(lla.flat(), [0, 0, 0])


    def test_ecef_to_lla_sym_case2(self):
        radius_equatorial = 6_378_137.0  # meters
        ecef = Matrix([radius_equatorial, 2000, 100])  # ecef @ t=0
        lla = ecef_to_lla_sym(ecef)
        truth = [0.000015784224246508,
                 0.000313571178299987,
                 0.31436039186129777]
        for i, j in zip(lla.flat(), truth):
            self.assertAlmostEqual(float(i), float(j), places=self.TOLERANCE_PLACES)


if __name__ == '__main__':
    unittest.main()
