from typing import Optional
import numpy as np
import dill as pickle
from japl.Util.Matlab import MatFile
from scipy.interpolate import RegularGridInterpolator



class Increments:
    burn_time = np.empty([])
    nburn_time = 0


    def get(self, name: str) -> np.ndarray:
        return self.__getattribute__(name)


    def __repr__(self) -> str:
        members = []
        for i in dir(self):
            attr = getattr(self, i)
            if isinstance(attr, np.ndarray):
                try:
                    members += [str(i) + f" [{len(attr)}]"]
                except Exception as e:  # noqa
                    members += [str(i) + " []"]
        return "\n".join(members)


    def __iter__(self):
        for k, v in self.__dict__.items():
            yield k, v


# TODO: do this better?
def from_CMS_table(data: MatFile|dict, units: str = "si") -> tuple[MatFile|dict, tuple]:

    convert_key_map = [("nozzle_area", "NozzleArea"),
                       ("mass_dot", "Mdot"),
                       ("cg", "CG"),
                       ("wet_mass", "WetMass"),
                       ("dry_mass", "DryMass"),
                       ("vac_flag", "VacFlag"),
                       ("thrust", "Thrust"),
                       ("prop_mass", "PropMass"),  # propellant
                       ("burn_time", "StageTime"),
                       ("burn_ratio", "BurnRatio")]

    # store to correct attribute name
    for map in convert_key_map:
        key_out, key_in = map
        table: np.ndarray = data.get(key_in)  # type:ignore
        setattr(data, key_out, table)
        if table is not None:
            delattr(data, key_in)

    # calculate burn ratio if not already defined
    if ("burn_ratio" not in data) or (data.get("burn_ratio") is None):
        wet_mass = data.get("wet_mass")
        dry_mass = data.get("dry_mass")
        prop_mass = data.get("prop_mass")
        burn_ratio = (np.array([wet_mass] * len(prop_mass)) - prop_mass  # type:ignore
                      - np.array([dry_mass] * len(prop_mass))) / (wet_mass - dry_mass)  # type:ignore
        setattr(data, "burn_ratio", burn_ratio)

    burn_time_axis = {"burn_time": data.get("burn_time")}
    axes = (burn_time_axis,)
    return (data, axes)


def from_default_table(data: MatFile|dict, units: str = "si") -> tuple[MatFile|dict, tuple]:
    raise Exception("not implemented.")


class MassPropTable:

    """This class is for containing Mass Properties data for a particular
    SimObject."""

    def __init__(self, data: str|dict|MatFile, from_template: str = "", units: str = "si") -> None:
        self.units = units

        # load table from dict or MatFile
        data_dict = {}
        if isinstance(data, str):
            self.__path = data
            if ".pickle" in self.__path:
                with open(self.__path, "rb") as f:
                    data_dict = pickle.load(f)
            elif ".mat" in self.__path:
                data_dict = MatFile(self.__path)

        # initial processing dependent on specific input table
        # formatting.
        match from_template.lower():
            case "cms":
                data_dict, table_axes = from_CMS_table(data_dict, units=units)
            case _:
                data_dict, table_axes = from_default_table(data_dict, units=units)

        _nozzle_area = data_dict.get("nozzle_area", None)
        _mass_dot = data_dict.get("mass_dot", None)
        _cg = data_dict.get("cg", None)
        _dry_mass = data_dict.get("dry_mass", None)
        _wet_mass = data_dict.get("wet_mass", None)
        _vac_flag = data_dict.get("vac_flag", None)
        _thrust = data_dict.get("thrust", None)
        _prop_mass = data_dict.get("prop_mass", None)
        _burn_time = data_dict.get("burn_time", None)

        self.nozzle_area = _nozzle_area
        self.mass_dot = RegularGridInterpolator((_burn_time,), _mass_dot)
        self.cg = RegularGridInterpolator((_burn_time,), _cg)
        self.dry_mass = _dry_mass
        self.wet_mass = _wet_mass
        self.vac_flag = _vac_flag
        self.thrust = RegularGridInterpolator((_burn_time,), _thrust)
        self.prop_mass = _prop_mass
        self.burn_time = _burn_time

        self.burn_time_max = self.burn_time.max()


    def get_mass_dot(self, t: float) -> float:
        if t >= self.burn_time_max:
            return 0.0
        else:
            return self.mass_dot([t])[0]


    def get_cg(self, t: float) -> float:
        if t >= self.burn_time_max:
            return self.cg([self.burn_time_max])[0]
        else:
            return self.cg([t])[0]


    def get_thrust(self, t: float) -> float:
        if t >= self.burn_time_max:
            return 0.0
        else:
            return self.thrust([t])[0]


    def __repr__(self) -> str:
        members = [i for i in dir(self) if "__" not in i]
        return "\n".join(members)
