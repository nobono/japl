import numpy as np
from ambiance import Atmosphere as AmbianceAtmosphere
from scipy.interpolate import RegularGridInterpolator
# from cached_interpolate import CachingInterpolant



class Atmosphere:


    def __init__(self) -> None:
        self._alts = np.linspace(0, 80_000, 80_001)
        self._atmos = AmbianceAtmosphere(self._alts)

        # copy arrays of ambiance Atmosphere class
        # for some reason accessing through the ambiance
        # class is very slow
        self._pressure = self._atmos.pressure.copy()
        self._density = self._atmos.density.copy()
        self._temperature_in_celsius = self._atmos.temperature_in_celsius.copy()
        self._speed_of_sound = self._atmos.speed_of_sound.copy()
        self._grav_accel = self._atmos.grav_accel.copy()

        self._pressure_interp = RegularGridInterpolator((self._alts,), self._pressure)
        self._density_interp = RegularGridInterpolator((self._alts,), self._density)
        self._temperature_in_celsius_interp = RegularGridInterpolator((self._alts,), self._temperature_in_celsius)
        self._speed_of_sound_interp = RegularGridInterpolator((self._alts,), self._speed_of_sound)
        self._grav_accel_interp = RegularGridInterpolator((self._alts,), self._grav_accel)

        self.modules = {
                "atmosphere_pressure": self.pressure,
                "atmosphere_density": self.density,
                "atmosphere_temperature": self.temperature,
                "atmosphere_speed_of_sound": self.speed_of_sound,
                "atmosphere_grav_accel": self.grav_accel,
                "atmosphere_dynamic_pressure": self.dynamic_pressure,
                }



    def pressure(self, alt: float|list|np.ndarray) -> float:
        alt = np.maximum(alt, 0)
        if hasattr(alt, "__len__"):
            return self._pressure_interp(alt)  # type:ignore
        else:
            return self._pressure_interp([alt])[0]  # type:ignore


    def density(self, alt: float|list|np.ndarray) -> float:
        alt = np.maximum(alt, 0)
        if hasattr(alt, "__len__"):
            return self._density_interp(alt)  # type:ignore
        else:
            return self._density_interp([alt])[0]  # type:ignore


    def temperature(self, alt: float|list|np.ndarray) -> float:
        alt = np.maximum(alt, 0)
        if hasattr(alt, "__len__"):
            return self._temperature_in_celsius_interp(alt)  # type:ignore
        else:
            return self._temperature_in_celsius_interp([alt])[0]  # type:ignore


    def speed_of_sound(self, alt: float|list|np.ndarray) -> float:
        alt = np.maximum(alt, 0)
        if hasattr(alt, "__len__"):
            return self._speed_of_sound_interp(alt)  # type:ignore
        else:
            return self._speed_of_sound_interp([alt])[0]  # type:ignore


    def grav_accel(self, alt: float|list|np.ndarray) -> float:
        alt = np.maximum(alt, 0)
        if hasattr(alt, "__len__"):
            return self._grav_accel_interp(alt)  # type:ignore
        else:
            return self._grav_accel_interp([alt])[0]  # type:ignore


    def dynamic_pressure(self, vel: float|list|np.ndarray, alt: float|list|np.ndarray) -> float:
        if hasattr(vel, "__len__"):
            vel_mag = np.linalg.norm(vel)
        else:
            vel_mag = vel
        return vel_mag**2 * self.density(alt) / 2  # type:ignore
