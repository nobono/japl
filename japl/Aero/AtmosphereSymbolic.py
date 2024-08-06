from japl import Atmosphere
from sympy import Function



class AtmosphereSymbolic:

    """This is the Symbolic mirror of the Atmosphere module
    which can be used for creating models from symblic expressions."""


    def __init__(self) -> None:
        self.atmosphere = Atmosphere()
        self.modules = {
                # "atmosphere": self.atmosphere,
                "atmosphere_pressure": self.atmosphere.pressure,
                "atmosphere_temperature": self.atmosphere.temperature,
                "atmosphere_speed_of_sound": self.atmosphere.speed_of_sound,
                "atmosphere_grav_accel": self.atmosphere.grav_accel,
                "atmosphere_dynamics_pressure": self.atmosphere.dynamic_pressure,
                }
        self.pressure = Function("atmosphere_pressure")
        self.density = Function("atmosphere_density`")
        self.temperature = Function("atmosphere_temperature")
        self.speed_of_sound = Function("atmosphere_speed_of_sound")
        self.grav_accel = Function("atmosphere_grav_accel")
        self.dynamics_pressure = Function("atmosphere_dynamics_pressure")
