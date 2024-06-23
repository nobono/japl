import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interpn
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


    def pressure(self, alt: float|list) -> float:
        # return np.interp(alt, self._alts, self._atmos.pressure)
        if hasattr(alt, "__len__"):
            return self._pressure[[round(i) for i in alt]] #type:ignore
        else:
            return self._pressure[round(alt)] #type:ignore

    
    def density(self, alt: float|list) -> float:
        # return np.interp(alt, self._alts, self._atmos.density)
        if hasattr(alt, "__len__"):
            return self._density[[round(i) for i in alt]] #type:ignore
        else:
            return self._density[round(alt)] #type:ignore


    def temperature(self, alt: float|list) -> float:
        # return np.interp(alt, self._alts, self._atmos.temperature)
        if hasattr(alt, "__len__"):
            return self._temperature_in_celsius[[round(i) for i in alt]] #type:ignore
        else:
            return self._temperature_in_celsius[round(alt)] #type:ignore
    

    def speed_of_sound(self, alt: float|list) -> float:
        # return np.interp(alt, self._alts, self._atmos.speed_of_sound)
        if hasattr(alt, "__len__"):
            return self._speed_of_sound[[round(i) for i in alt]] #type:ignore
        else:
            return self._speed_of_sound[round(alt)] #type:ignore


    def grav_accel(self, alt: float|list) -> float:
        # return np.interp(alt, self._alts, self._atmos.grav_accel)
        if hasattr(alt, "__len__"):
            return self._grav_accel[[round(i) for i in alt]] #type:ignore
        else:
            return self._grav_accel[round(alt)] #type:ignore


    def dynamic_pressure(self, vel: float|np.ndarray, alt: float) -> float:
        if hasattr(vel, "__len__"):
            return np.linalg.norm(vel)**2 * self.density(alt) / 2 #type:ignore
        else:
            return vel**2 * self.density(alt) / 2 #type:ignore


