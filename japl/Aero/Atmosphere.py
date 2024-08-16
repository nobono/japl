import numpy as np
from ambiance import Atmosphere as AmbianceAtmosphere
# from cached_interpolate import CachingInterpolant



class Atmosphere:


    def __init__(self) -> None:
        self._alts = np.linspace(0, 80_000, 80_001)
        self._atmos = AmbianceAtmosphere(self._alts)

        # copy arrays of ambiance Atmosphere class
        # for some reason accessing through the ambiance
        # class is very slow
        self._pressure = self._atmos.speed_of_sound.copy()
        self._density = self._atmos.density.copy()
        self._temperature_in_celsius = self._atmos.temperature_in_celsius.copy()
        self._speed_of_sound = self._atmos.speed_of_sound.copy()
        self._grav_accel = self._atmos.grav_accel.copy()


    def pressure(self, alt: float|list|np.ndarray) -> float:
        if hasattr(alt, "__len__"):
            return self._pressure[[round(i) for i in alt]]  # type:ignore
        else:
            return self._pressure[round(alt)]  # type:ignore


    def density(self, alt: float|list|np.ndarray) -> float:
        if hasattr(alt, "__len__"):
            return self._density[[round(i) for i in alt]]  # type:ignore
        else:
            return self._density[round(alt)]  # type:ignore


    def temperature(self, alt: float|list|np.ndarray) -> float:
        if hasattr(alt, "__len__"):
            return self._temperature_in_celsius[[round(i) for i in alt]]  # type:ignore
        else:
            return self._temperature_in_celsius[round(alt)]  # type:ignore


    def speed_of_sound(self, alt: float|list|np.ndarray) -> float:
        if hasattr(alt, "__len__"):
            return self._speed_of_sound[[round(i) for i in alt]]  # type:ignore
        else:
            return self._speed_of_sound[round(alt)]  # type:ignore


    def grav_accel(self, alt: float|list|np.ndarray) -> float:
        if hasattr(alt, "__len__"):
            return self._grav_accel[[round(i) for i in alt]]  # type:ignore
        else:
            return self._grav_accel[round(alt)]  # type:ignore


    def dynamic_pressure(self, vel: float|list|np.ndarray, alt: float|list|np.ndarray) -> float:
        if hasattr(vel, "__len__"):
            vel_mag = np.linalg.norm(vel)
        else:
            vel_mag = vel
        return vel_mag**2 * self.density(alt) / 2  # type:ignore
