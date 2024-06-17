import numpy as np
# from scipy.interpolate import interp1d
from ambiance import Atmosphere as AmbianceAtmosphere
# from cached_interpolate import CachingInterpolant



class Atmosphere:


    def __init__(self) -> None:
        self._alts = np.linspace(0, 80_000, 80_001)
        self._atmos = AmbianceAtmosphere(self._alts)
        # self._pressure = CachingInterpolant(
        #     x=self._alts,
        #     y=self._atmos.pressure,
        # )
        # self._density = CachingInterpolant(
        #     x=self._alts,
        #     y=self._atmos.density,
        # )
        # self._temperature = CachingInterpolant(
        #     x=self._alts,
        #     y=self._atmos.temperature,
        # )
        # self._speed_of_sound = CachingInterpolant(
        #     x=self._alts,
        #     y=self._atmos.speed_of_sound,
        # )
        # self._grav_accel = CachingInterpolant(
        #     x=self._alts,
        #     y=self._atmos.grav_accel,
        # )
        # self._density = interp1d(self._alts, self._atmos.density)
        # self._temperature = interp1d(self._alts, self._atmos.temperature)
        # self._speed_of_sound = interp1d(self._alts, self._atmos.speed_of_sound)
        # self._grav_accel = interp1d(self._alts, self._atmos.grav_accel)


    def pressure(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.pressure)
        return self._atmos.pressure[round(alt)]

    
    def density(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.density)
        return self._atmos.density[round(alt)]


    def temperature(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.temperature)
        return self._atmos.temperature_in_celsius[round(alt)]
    

    def speed_of_sound(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.speed_of_sound)
        return self._atmos.speed_of_sound[round(alt)]


    def grav_accel(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.grav_accel)
        return self.grav_accel(alt)


    def dynamic_pressure(self, vel: np.ndarray, alt: float) -> float:
        if isinstance(vel, np.ndarray):
            return np.linalg.norm(vel) * self.density(alt) / 2 #type:ignore
        elif isinstance(vel, float):
            return vel * self.density(alt) / 2


