from japl import Atmosphere
from sympy import Function



class AtmosphereSymbolic:

    """This is the Symbolic mirror of the Atmosphere module
    which can be used for creating models from symblic expressions."""


    def __init__(self) -> None:
        self.atmosphere = Atmosphere()
        self.modules = {
                # "atmosphere": self.atmosphere,
                "atmosphere.pressure": self.atmosphere.pressure,
                "atmosphere.density": self.atmosphere.density,
                "atmosphere.temperature": self.atmosphere.temperature,
                "atmosphere.speed_of_sound": self.atmosphere.speed_of_sound,
                "atmosphere.grav_accel": self.atmosphere.grav_accel,
                "atmosphere.dynamic_pressure": self.atmosphere.dynamic_pressure,
                }
        self.pressure = Function("atmosphere.pressure")
        self.density = Function("atmosphere.density")
        self.temperature = Function("atmosphere.temperature")
        self.speed_of_sound = Function("atmosphere.speed_of_sound")
        self.grav_accel = Function("atmosphere.grav_accel")
        self.dynamic_pressure = Function("atmosphere.dynamic_pressure")
