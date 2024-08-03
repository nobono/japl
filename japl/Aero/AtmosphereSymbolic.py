from japl import Atmosphere
from sympy import Function



class AtmosphereSymbolic:

    """This is the Symbolic mirror of the Atmosphere module
    which can be used for creating models from symblic expressions."""


    def __init__(self) -> None:
        self.atmosphere = Atmosphere()
        self.modules = {
                "atmosphere": self.atmosphere,
                "pressure": self.atmosphere.pressure,
                "temperature": self.atmosphere.temperature,
                "speed_of_sound": self.atmosphere.speed_of_sound,
                "grav_accel": self.atmosphere.grav_accel,
                "dynamics_pressure": self.atmosphere.dynamic_pressure,
                }
        self.pressure = Function("atmosphere.pressure")
        self.density = Function("atmosphere.density`")
        self.temperature = Function("atmosphere.temperature")
        self.speed_of_sound = Function("atmosphere.speed_of_sound")
        self.grav_accel = Function("atmosphere.grav_accel")
        self.dynamics_pressure = Function("atmosphere.dynamics_pressure")
