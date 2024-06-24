import unittest
import numpy as np
import quaternion
from japl.Math.Rotation import quat_to_dcm
from japl.Math.Rotation import quat_to_tait_bryan
from japl.Math.Rotation import tait_bryan_to_dcm
from japl.Math.Rotation import dcm_to_tait_bryan



class TestMathRotation(unittest.TestCase):


    def setUp(self) -> None:
        self.EPSILON = 1e-12


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


if __name__ == '__main__':
    unittest.main()
