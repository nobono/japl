import numpy as np
from scipy.interpolate import interp1d
from ambiance import Atmosphere as AmbianceAtmosphere
from cached_interpolate import CachingInterpolant



class Atmosphere:


    def __init__(self) -> None:
        self._alts = np.linspace(0, 80_000, 1000)
        self._atmos = AmbianceAtmosphere(self._alts)
        self._pressure = CachingInterpolant(
            x=self._alts,
            y=self._atmos.pressure,
        )
        self._density = CachingInterpolant(
            x=self._alts,
            y=self._atmos.density,
        )
        self._temperature = CachingInterpolant(
            x=self._alts,
            y=self._atmos.temperature,
        )
        self._speed_of_sound = CachingInterpolant(
            x=self._alts,
            y=self._atmos.speed_of_sound,
        )
        self._grav_accel = CachingInterpolant(
            x=self._alts,
            y=self._atmos.grav_accel,
        )
        # self._density = interp1d(self._alts, self._atmos.density)
        # self._temperature = interp1d(self._alts, self._atmos.temperature)
        # self._speed_of_sound = interp1d(self._alts, self._atmos.speed_of_sound)
        # self._grav_accel = interp1d(self._alts, self._atmos.grav_accel)


    def pressure(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.pressure)
        return self._pressure(alt)

    
    def density(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.density)
        return self._density(alt)


    def temperature(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.temperature)
        return self._temperature(alt)
    

    def speed_of_sound(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.speed_of_sound)
        return self._speed_of_sound(alt)


    def grav_accel(self, alt: float) -> float:
        # return np.interp(alt, self._alts, self._atmos.grav_accel)
        return self._grav_accel(alt)