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
        if (wet_mass - dry_mass) == 0.0:
            burn_ratio = np.zeros_like(wet_mass)
        else:
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

    def __init__(self, data: Optional[str|dict|MatFile] = None, from_template: str = "", units: str = "si") -> None:
        self.units = units
        self.stages: list[MassPropTable] = []
        self.is_stage: bool = True
        self.stage_id: int = 0

        # load table from dict or MatFile
        data_dict = {}
        if data is None:
            return
        elif isinstance(data, str):
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

        self._nozzle_area = data_dict.get("nozzle_area", None)
        self._mass_dot = data_dict.get("mass_dot", None)
        self._cg = data_dict.get("cg", None)
        self._dry_mass = data_dict.get("dry_mass", None)
        self._wet_mass = data_dict.get("wet_mass", None)
        self._vac_flag = data_dict.get("vac_flag", None)
        self._thrust = data_dict.get("thrust", None)
        self._prop_mass = data_dict.get("prop_mass", None)
        self._burn_time = data_dict.get("burn_time", None)

        self.nozzle_area: float = self._nozzle_area
        self.mass_dot = RegularGridInterpolator((self._burn_time,), self._mass_dot)
        self.cg = RegularGridInterpolator((self._burn_time,), self._cg)
        self.dry_mass = self._dry_mass
        self.wet_mass = self._wet_mass
        self.vac_flag = self._vac_flag
        self.thrust = RegularGridInterpolator((self._burn_time,), self._thrust)
        self.prop_mass = self._prop_mass
        self.burn_time = self._burn_time

        self.burn_time_max = self.burn_time.max()


    @DeprecationWarning
    def set(self, mass_props: "MassPropTable") -> None:
        """This method re-initializes the mass-property tables with the
        provided MassPropTable argument. This is to provide different
        tables if a model switches stages."""
        table_names = ["nozzle_area",
                       "mass_dot",
                       "cg",
                       "dry_mass",
                       "wet_mass",
                       "vac_flag",
                       "thrust",
                       "prop_mass",
                       "burn_time",
                       "burn_time_max",
                       ]
        for name in table_names:
            setattr(self, name, getattr(mass_props, name))


    def add_stage(self, child) -> None:
        """Adds a child object as an ordered child
        of this object."""
        self.is_stage = False
        child.is_stage = True
        self.stages += [child]

    def set_stage(self, stage: int) -> None:
        """Set the current stage index for the object. This is
        so that `get_stage()` will return the corresponding aerotable."""
        if int(stage) >= len(self.stages):
            raise Exception(f"cannot access stage {int(stage)} "
                            f"for container of size {len(self.stages)}")
        self.stage_id = int(stage)

    def get_stage(self):
        """Returns the current object corresponding to the stage_id."""
        if self.is_stage:
            return self
        else:
            return self.stages[self.stage_id]


    def get_wet_mass(self) -> float:
        stage = self.get_stage()
        return stage.wet_mass


    def get_dry_mass(self) -> float:
        stage = self.get_stage()
        return stage.dry_mass


    def get_mass_dot(self, t: float) -> float:
        stage = self.get_stage()
        if t >= stage.burn_time_max:
            return 0.0
        else:
            return stage.mass_dot([t])[0]


    def get_cg(self, t: float) -> float:
        stage = self.get_stage()
        if t >= stage.burn_time_max:
            return stage.cg([stage.burn_time_max])[0]
        else:
            return stage.cg([t])[0]


    def get_isp(self, t: float, pressure: float) -> float:
        stage = self.get_stage()
        thrust = stage.get_thrust(t, pressure)
        g0 = 9.80665
        mass_dot = stage.get_mass_dot(t)
        isp = thrust / (mass_dot * g0)
        return isp


    def get_raw_thrust(self, t: float) -> float:
        """(vac_thrust)"""
        stage = self.get_stage()
        if t >= stage.burn_time_max:
            return 0.0
        else:
            return stage.thrust([t])[0]


    def get_thrust(self, t: float, pressure: float):
        stage = self.get_stage()
        if t <= stage.burn_time_max:
            raw_thrust = self.get_raw_thrust(t)
            if stage.vac_flag:
                vac_thrust = raw_thrust
                thrust = max(vac_thrust - np.sign(vac_thrust) * stage.nozzle_area * pressure, 0)
            else:
                thrust = raw_thrust
                vac_thrust = thrust + stage.nozzle_area * pressure
            return thrust
        else:
            return 0


    def __repr__(self) -> str:
        members = [i for i in dir(self) if "__" not in i]
        return "\n".join(members)
