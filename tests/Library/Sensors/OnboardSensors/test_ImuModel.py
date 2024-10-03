import unittest
import numpy as np
from japl.Library.Sensors.OnboardSensors.ImuModel import SensorBase
from japl.Library.Sensors.OnboardSensors.ImuModel import ImuSensor
from japl.Math.Rotation import Sq
from japl.Math.Rotation import quat_norm
from japl.Math.Rotation import quat_to_dcm
from japl.Math.Vec import vec_ang



class TestSensorBase3Dof(unittest.TestCase):


    def setUp(self) -> None:
        self.TOLERANCE_PLACES = 15
        np.random.seed(123)
        np.set_printoptions(precision=18)


    @staticmethod
    def model(vec: np.ndarray|list,
              quat: np.ndarray|list,
              ang_vel: np.ndarray|list,
              dt: float):
        """quaternion rotation model."""
        vec = np.asarray(vec)
        quat = np.asarray(quat)
        ang_vel = np.asarray(ang_vel)
        quat_dot = 0.5 * Sq(quat) @ ang_vel
        quat_new = quat + quat_dot * dt
        quat_new = quat_norm(quat_new)
        dcm_new = quat_to_dcm(quat_new)
        return dcm_new @ vec


    def test_base_case1(self):
        sensor = SensorBase(scale_factor=np.ones(3),
                            misalignment=np.zeros(3),
                            bias=np.zeros(3),
                            noise=np.zeros(3),
                            delay=0)

        true = np.array([[0.0997506234413965, 0., 0.9950124688279302],
                         [0.19850622819509828, 0., 0.9800996262461054],
                         [0.2952817209468539, 0., 0.9554102287890077]])

        grav_vec0 = np.array([0, 0, 1])
        dt = 0.1
        quat = [1, 0, 0, 0]
        ang_vel = [0, 1, 0]
        grav_vec = grav_vec0
        for istep, t in enumerate(np.arange(0, 0.3, dt)):
            grav_vec = self.model(grav_vec, quat, ang_vel, dt)
            ret = sensor.get_measurement(t, grav_vec)
            self.assertListEqual(true[istep].tolist(), ret.tolist())


    def test_base_scalefactor(self):
        sensor = SensorBase(scale_factor=np.ones(3) * 2,
                            misalignment=np.zeros(3),
                            bias=np.zeros(3),
                            noise=np.zeros(3),
                            delay=0)

        true = np.array([[0.0997506234413965, 0., 0.9950124688279302],
                         [0.19850622819509828, 0., 0.9800996262461054],
                         [0.2952817209468539, 0., 0.9554102287890077]])

        grav_vec0 = np.array([0, 0, 1])
        dt = 0.1
        quat = [1, 0, 0, 0]
        ang_vel = [0, 1, 0]
        grav_vec = grav_vec0
        for istep, t in enumerate(np.arange(0, 0.3, dt)):
            grav_vec = self.model(grav_vec, quat, ang_vel, dt)
            ret = sensor.get_measurement(t, grav_vec)
            # ang = np.degrees(vec_ang(grav_vec, grav_vec0))
            self.assertListEqual((true[istep] * 2).tolist(), ret.tolist())


    def test_base_misalign(self):
        sensor = SensorBase(scale_factor=np.ones(3),
                            misalignment=[.1, .1, .1],
                            bias=np.zeros(3),
                            noise=np.zeros(3),
                            delay=0)

        grav_vec0 = np.array([0, 0, 1])
        dt = 0.1
        quat = [1, 0, 0, 0]
        ang_vel = [0, 0, 0]
        grav_vec = grav_vec0
        for istep, t in enumerate(np.arange(0, 0.3, dt)):
            grav_vec = self.model(grav_vec, quat, ang_vel, dt)
            ret = sensor.get_measurement(t, grav_vec)
            self.assertListEqual([.1, .1, 1], ret.tolist())


    def test_base_bias(self):
        sensor = SensorBase(scale_factor=np.ones(3),
                            misalignment=np.zeros(3),
                            bias=[2, 3, 0],
                            noise=np.zeros(3),
                            delay=0)

        grav_vec0 = np.array([0, 0, 1])
        dt = 0.1
        quat = [1, 0, 0, 0]
        ang_vel = [0, 0, 0]
        grav_vec = grav_vec0
        for istep, t in enumerate(np.arange(0, 0.3, dt)):
            grav_vec = self.model(grav_vec, quat, ang_vel, dt)
            ret = sensor.get_measurement(t, grav_vec)
            self.assertListEqual([2, 3, 1], ret.tolist())


    def test_base_noise(self):
        sensor = SensorBase(scale_factor=np.ones(3),
                            misalignment=np.zeros(3),
                            bias=np.zeros(3),
                            noise=[1, 0, 0],
                            delay=0)

        true = np.array([[0.3929383711957233, 0., 1.],
                         [0.10262953816578246, 0., 1.],
                         [0.961528396769231, 0., 1.]])

        grav_vec0 = np.array([0, 0, 1])
        dt = 0.1
        quat = [1, 0, 0, 0]
        ang_vel = [0, 0, 0]
        grav_vec = grav_vec0
        for istep, t in enumerate(np.arange(0, 0.3, dt)):
            grav_vec = self.model(grav_vec, quat, ang_vel, dt)
            ret = sensor.get_measurement(t, grav_vec)
            self.assertListEqual(true[istep].tolist(), ret.tolist())

    ##################################################################
    # IMU tests
    ##################################################################

    def test_imu_case1(self):
        pass


if __name__ == '__main__':
    unittest.main()
