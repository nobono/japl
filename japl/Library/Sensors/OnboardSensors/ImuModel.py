import numpy as np
from typing import Union
from japl.Math.Rotation import Sq
from japl.Math.Rotation import quat_norm
from japl.Math.Rotation import quat_to_dcm
from japl.Sim.Integrate import runge_kutta_4
from japl.Sim.Integrate import euler
# from collections import deque
from queue import Queue

IterT = Union[np.ndarray, list]


class Measurement:


    def __init__(self, timestamp: float, value: np.ndarray) -> None:
        self.timestamp = timestamp
        self.value = value


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
        self.buffer = Queue()
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


    def count(self) -> int:
        return self.buffer.qsize()


    def get_noise(self) -> np.ndarray:
        """returns random uniform noise array reflective
        of the self.noise parameter."""
        return np.random.normal([0, 0, 0], self.noise)


    def calc_measurement(self, time: float, true_val: np.ndarray):
        return self.M @ self.S @ true_val + self.bias + self.get_noise()


    def update(self, time: float, true_val: np.ndarray):
        meas = self.calc_measurement(time=time, true_val=true_val)
        buf_info = Measurement(time, meas)
        self.buffer.put(buf_info)


    def get_measurement(self, nget: int = -1) -> list[Measurement]:
        if nget < 0:
            nget = self.buffer.qsize()
        return [self.buffer.get() for i in range(nget)]


    def get_latest_measurement(self) -> Measurement:
        return self.get_measurement()[-1]  # type:ignore


class ImuSensor:

    def __init__(self,
                 accel: SensorBase = SensorBase(),
                 gyro: SensorBase = SensorBase(),
                 mag: SensorBase = SensorBase()) -> None:
        self.accelerometer = accel
        self.gyroscope = gyro
        self.magnetometer = mag


    @staticmethod
    def _rotation_dynamics(t, X, *args):
        """for dev purposes"""
        quat = X[:4]
        ang_vel = args[0]
        return 0.5 * Sq(quat) @ ang_vel


    def _rotation_model(self,
                        quat: np.ndarray,
                        ang_vel: np.ndarray,
                        dt: float):
        """for dev purposes"""
        quat = np.asarray(quat)
        ang_vel = np.asarray([0, 1, 0])
        quat_new, _ = runge_kutta_4(f=self._rotation_dynamics, t=0, X=quat, dt=dt, args=(ang_vel,))
        # quat_new, _ = euler(self.rotation_dynamics, t=0, X=quat, dt=dt, args=(ang_vel,))
        quat_new = quat_norm(quat_new)
        return quat_new


    def update(self,
               time: float,
               acc_vec: np.ndarray|list,
               ang_vel: np.ndarray|list,
               mag_vec: np.ndarray|list) -> None:
        acc_vec = np.asarray(acc_vec)
        ang_vel = np.asarray(ang_vel)
        mag_vec = np.asarray(mag_vec)
        self.accelerometer.update(time, acc_vec)
        self.gyroscope.update(time, ang_vel)
        self.magnetometer.update(time, mag_vec)
