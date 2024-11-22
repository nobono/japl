from japl import Atmosphere
# from sympy import Function
from sympy import Symbol
from japl.Symbolic.KwargFunction import KwargFunction



class pressure(KwargFunction):
    pass
    parent = "atmosphere"
    # no_keyword = True


class density(KwargFunction):
    pass
    parent = "atmosphere"
    # no_keyword = True


class temperature(KwargFunction):
    pass
    parent = "atmosphere"
    # no_keyword = True


class speed_of_sound(KwargFunction):
    pass
    parent = "atmosphere"
    # no_keyword = True


class grav_accel(KwargFunction):
    pass
    parent = "atmosphere"
    # no_keyword = True


class dynamic_pressure(KwargFunction):
    pass
    parent = "atmosphere"
    # no_keyword = True


class AtmosphereSymbolic:

    """This is the Symbolic mirror of the Atmosphere module
    which can be used for creating models from symblic expressions."""


    def __init__(self) -> None:
        # self.atmosphere = Atmosphere()
        # self.modules = {
        #         # "atmosphere": self.atmosphere,
        #         "atmosphere.pressure": self.atmosphere.pressure,
        #         "atmosphere.density": self.atmosphere.density,
        #         "atmosphere.temperature": self.atmosphere.temperature,
        #         "atmosphere.speed_of_sound": self.atmosphere.speed_of_sound,
        #         "atmosphere.grav_accel": self.atmosphere.grav_accel,
        #         "atmosphere.dynamic_pressure": self.atmosphere.dynamic_pressure,
        #         }
        # atmosphere = Symbol("atmosphere")
        # self.pressure = Function("atmosphere_pressure")
        # self.density = Function("atmosphere_density")
        # self.temperature = Function("atmosphere.temperature")
        # self.speed_of_sound = Function("atmosphere.speed_of_sound")
        # self.grav_accel = Function("atmosphere.grav_accel")
        # self.dynamic_pressure = Function("atmosphere.dynamic_pressure")
        self.modules = {}
        self.pressure = pressure
        self.density = density
        self.temperature = temperature
        self.speed_of_sound = speed_of_sound
        self.grav_accel = grav_accel
        self.dynamic_pressure = dynamic_pressure
