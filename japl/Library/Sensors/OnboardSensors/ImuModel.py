import numpy as np
# from japl import Model
# from sympy import symbols, Matrix, lambdify



class SensorBase:


    def __init__(self, scale_factor, misalignment, bias, noise, delay, dof: int = 3) -> None:
        assert len(scale_factor) == len(misalignment) == len(bias)
        assert dof >= len(scale_factor)
        self.dof = dof
        self.scale_factor = scale_factor
        self.misalignment = misalignment
        self.bias = bias
        self.noise = noise
        self.noise_func = lambda: np.array([np.random.uniform(-noise[i], noise[i]) for i in range(self.dof)])
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


    def _calc_measurement(self, time, true_val):
        return self.M @ self.S @ true_val + self.bias + self.noise


    def get_measurement(self, time, true_val):
        self._calc_measurement(time, true_val)
