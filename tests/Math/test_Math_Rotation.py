import unittest
import numpy as np
import quaternion
from japl.Math.Rotation import quat_conj
from japl.Math.Rotation import quat_to_dcm
from japl.Math.Rotation import quat_to_tait_bryan
from japl.Math.Rotation import tait_bryan_to_dcm
from japl.Math.Rotation import dcm_to_tait_bryan
from sympy import MatrixSymbol
from japl.Math.RotationSymbolic import quat_conj_sym
from japl.Math.RotationSymbolic import quat_to_dcm_sym
from japl.Math.RotationSymbolic import quat_to_tait_bryan_sym
from japl.Math.RotationSymbolic import tait_bryan_to_dcm_sym
from japl.Math.RotationSymbolic import dcm_to_tait_bryan_sym



class TestMathRotation(unittest.TestCase):


    def setUp(self) -> None:
        self.EPSILON = 1e-15


    def test_quat_conj(self):
        quat = quaternion.from_euler_angles(0, 0, np.radians(45))
        conj = quat_conj(quat.components)
        self.assertTrue(quat.w == conj[0])
        self.assertTrue((quat.components[1:] == -conj[1:]).all())


    def test_quat_to_dcm(self):
        # quat = quaternion.from_euler_angles(*np.radians([10, 20, 30]))
        quat = quaternion.from_float_array([0.925416578398323364, 0.030153689607045796, 0.171010071662834329, 0.336824088833465152])
        dcm = quat_to_dcm(quat)
        truth = np.array([[0.7146101771427564, -0.6130920223795969, 0.3368240888334651],
                          [0.633718360861996, 0.7712805763691759, 0.0593911746138847],
                          [-0.29619813272602374, 0.17101007166283433, 0.9396926207859084]])
        self.assertTrue((dcm == truth).all())


    def test_tait_bryan_to_dcm(self):
        yaw_pitch_roll = np.radians([10, 20, 30])
        dcm = tait_bryan_to_dcm(yaw_pitch_roll)
        truth = np.array([[0.925416578398323364, 0.018028311236297251, 0.378522306369792449],
                  [0.163175911166534821, 0.882564119259385604, -0.440969610529882372],
                  [-0.342020143325668713, 0.469846310392954158, 0.813797681349373803]])
        self.assertTrue((dcm == truth).all())


    def test_dcm_to_tait_bryan(self):
        # quat = quaternion.from_euler_angles(*np.radians([10, 20, 30]))
        quat = quaternion.from_float_array([0.925416578398323364, 0.030153689607045796, 0.171010071662834329, 0.336824088833465152])
        dcm = quaternion.as_rotation_matrix(quat)
        tb_angles = dcm_to_tait_bryan(dcm)
        truth = np.array([0.725475843410945509, 0.300709698155427141, 0.180015088428340159])
        self.assertTrue((tb_angles == truth).all())


    def test_quat_to_tait_bryan(self):
        quat = quaternion.from_float_array([0.925416578398323364, 0.030153689607045796, 0.171010071662834329, 0.336824088833465152])
        tb_angles = quat_to_tait_bryan(quat)
        truth = np.array([0.725475843410945509, 0.300709698155427141, 0.180015088428340159])
        self.assertTrue((tb_angles == truth).all())


    def __angles_to_dcm_to_quat_to_dcm_to_angles(self, tait_bryan_angles):
        dcm = tait_bryan_to_dcm(tait_bryan_angles)
        q = quaternion.from_rotation_matrix(dcm)
        dcm = quat_to_dcm(q)
        tb_angles = dcm_to_tait_bryan(dcm)
        return tb_angles


    def test_quat_to_tait_bryan_case1(self):
        tait_bryan_angles = np.radians([0, 45, 0])
        tb_angles_out = self.__angles_to_dcm_to_quat_to_dcm_to_angles(tait_bryan_angles)
        self.assertTrue(abs(tb_angles_out[0] - tait_bryan_angles[0]) < self.EPSILON)
        self.assertTrue(abs(tb_angles_out[1] - tait_bryan_angles[1]) < self.EPSILON)
        self.assertTrue(abs(tb_angles_out[2] - tait_bryan_angles[2]) < self.EPSILON)

        tait_bryan_angles = np.radians([45, 0, 0])
        tb_angles_out = self.__angles_to_dcm_to_quat_to_dcm_to_angles(tait_bryan_angles)
        self.assertTrue(abs(tb_angles_out[0] - tait_bryan_angles[0]) < self.EPSILON)
        self.assertTrue(abs(tb_angles_out[1] - tait_bryan_angles[1]) < self.EPSILON)
        self.assertTrue(abs(tb_angles_out[2] - tait_bryan_angles[2]) < self.EPSILON)

        tait_bryan_angles = np.radians([0, 0, 45])
        tb_angles_out = self.__angles_to_dcm_to_quat_to_dcm_to_angles(tait_bryan_angles)
        self.assertTrue(abs(tb_angles_out[0] - tait_bryan_angles[0]) < self.EPSILON)
        self.assertTrue(abs(tb_angles_out[1] - tait_bryan_angles[1]) < self.EPSILON)
        self.assertTrue(abs(tb_angles_out[2] - tait_bryan_angles[2]) < self.EPSILON)


    def test_tait_bryan_angles_case2(self):
        tait_bryan_angles = np.radians([10, 90, 5])
        tb_angles_out = self.__angles_to_dcm_to_quat_to_dcm_to_angles(tait_bryan_angles)
        tb_angles_out = np.degrees(tb_angles_out)
        self.assertTrue(abs(tb_angles_out[0] - 175.0) < self.EPSILON)
        self.assertTrue(abs(tb_angles_out[1] - 90.0) < self.EPSILON)
        self.assertTrue(abs(tb_angles_out[2] - 0.0) < self.EPSILON)


    ###################################################################################
    # Symbolic
    ###################################################################################

    def test_quat_conj_sym(self):
        quat = MatrixSymbol('q', 4, 1)
        conj = quat_conj_sym(quat)
        self.assertTrue(quat[0] == conj[0])
        self.assertTrue(quat[1] == -conj[1]) #type:ignore
        self.assertTrue(quat[2] == -conj[2]) #type:ignore
        self.assertTrue(quat[3] == -conj[3]) #type:ignore


    def test_quat_to_dcm_sym(self):
        quat = MatrixSymbol("q", 4, 1).as_mutable()
        dcm = quat_to_dcm_sym(quat)
        subs = {quat[0]: 0.925416578398323364, #type:ignore
                quat[1]: 0.030153689607045796, #type:ignore
                quat[2]: 0.171010071662834329, #type:ignore
                quat[3]: 0.336824088833465152} #type:ignore
        truth = np.array([[0.7146101771427564, -0.6130920223795969, 0.3368240888334651],
                          [0.633718360861996, 0.7712805763691759, 0.0593911746138847],
                          [-0.29619813272602374, 0.17101007166283433, 0.9396926207859084]])
        dcm_subbed = np.array(dcm.subs(subs)).squeeze()
        self.assertTrue((dcm_subbed == truth).all())


    def test_tait_bryan_to_dcm_sym(self):
        yaw_pitch_roll = MatrixSymbol("yaw_pitch_roll", 3, 1).as_mutable()
        dcm = tait_bryan_to_dcm_sym(yaw_pitch_roll)
        subs = {yaw_pitch_roll[0]: np.radians(10), #type:ignore
                yaw_pitch_roll[1]: np.radians(20), #type:ignore
                yaw_pitch_roll[2]: np.radians(30)} #type:ignore
        truth = np.array([[0.925416578398323364, 0.018028311236297251, 0.378522306369792449],
                  [0.163175911166534821, 0.882564119259385604, -0.440969610529882372],
                  [-0.342020143325668713, 0.469846310392954158, 0.813797681349373803]])
        dcm_subbed = np.array(dcm.subs(subs)).squeeze()
        self.assertTrue((dcm_subbed == truth).all())


    def test_dcm_to_tait_bryan_sym(self):
        quat = MatrixSymbol("q", 4, 1).as_mutable()
        dcm = quat_to_dcm_sym(quat)
        subs = {quat[0]: 0.925416578398323364, #type:ignore
                quat[1]: 0.030153689607045796, #type:ignore
                quat[2]: 0.171010071662834329, #type:ignore
                quat[3]: 0.336824088833465152} #type:ignore
        tb_angles = dcm_to_tait_bryan_sym(dcm)
        truth = np.array([0.725475843410945509,
                          0.300709698155427141,
                          0.180015088428340159])
        tb_angles_subbed = np.array(tb_angles.subs(subs)).squeeze()
        self.assertTrue((tb_angles_subbed == truth).all())


    def test_quat_to_tait_bryan_sym(self):
        quat = MatrixSymbol("q", 4, 1).as_mutable()
        subs = {quat[0]: 0.925416578398323364, #type:ignore
                quat[1]: 0.030153689607045796, #type:ignore
                quat[2]: 0.171010071662834329, #type:ignore
                quat[3]: 0.336824088833465152} #type:ignore
        tb_angles = quat_to_tait_bryan_sym(quat)
        truth = np.array([0.725475843410945509,
                          0.300709698155427141,
                          0.180015088428340159])
        tb_angles_subbed = np.array(tb_angles.subs(subs)).squeeze()
        self.assertTrue((tb_angles_subbed == truth).all())


if __name__ == '__main__':
    unittest.main()