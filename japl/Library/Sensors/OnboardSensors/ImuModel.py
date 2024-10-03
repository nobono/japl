import numpy as np
from typing import Union
from japl.Math.Rotation import Sq
from japl.Math.Rotation import quat_norm
from japl.Math.Rotation import quat_to_dcm

IterT = Union[np.ndarray, list]



class SensorBase:


    def __init__(self,
                 scale_factor: IterT = np.ones(3),
                 misalignment: IterT = np.zeros(3),
                 bias: IterT = np.zeros(3),
                 noise: IterT = np.zeros(3),
                 delay: float = 0,
                 dof: int = 3) -> None:
        assert len(scale_factor) == len(misalignment) == len(bias)
        assert dof >= len(scale_factor)
        self.dof = dof
        self.scale_factor = np.asarray(scale_factor)
        self.misalignment = np.asarray(misalignment)
        self.bias = np.asarray(bias)
        self.noise = np.asarray(noise)
        self.delay = delay
        self.last_measurement_time = 0
        self.measurement_buffer = []
        self.S = np.array([
            [scale_factor[0], 0, 0],
            [0, scale_factor[1], 0],
            [0, 0, scale_factor[2]],
            ])
        self.M = np.array([
            [1, misalignment[0], misalignment[1]],
            [misalignment[0], 1, misalignment[2]],
            [misalignment[1], misalignment[2], 1],
            ])


    def get_noise(self):
        """returns random uniform noise array reflective
        of the self.noise parameter."""
        return np.array([np.random.uniform(-i, i) for i in self.noise])


    def get_measurement(self, time: float, true_val: np.ndarray):
        return self.M @ self.S @ true_val + self.bias + self.get_noise()


class ImuSensor:

    def __init__(self,
                 accel: SensorBase = SensorBase(),
                 gyro: SensorBase = SensorBase(),
                 mag: SensorBase = SensorBase()) -> None:
        self.accelerometer = accel
        self.gyroscope = gyro
        self.magnetometer = mag


    @staticmethod
    def rotation_model(grav_vec: np.ndarray,
                       ang_vel: np.ndarray,
                       mag_vec: np.ndarray,
                       quat: np.ndarray,
                       dt: float):
        # vec = np.asarray(vec)
        quat = np.asarray(quat)
        ang_vel = np.asarray(ang_vel)
        quat_dot = 0.5 * Sq(quat) @ ang_vel
        quat_new = quat + quat_dot * dt
        quat_new = quat_norm(quat_new)
        dcm_new = quat_to_dcm(quat_new)
        return dcm_new @ vec
