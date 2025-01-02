from typing import Optional
import numpy as np
from pathlib import Path
from astropy.units import Unit
from astropy import units as u
from japl.DataTable.DataTable import DataTable
from japl.Util.Matlab import MatFile
from japl.Util.Staged import Staged
from masstable import MassTable as CppMassTable



class MassTable(Staged):

    """This class is for containing Mass Properties data for a particular
    SimObject."""

    nozzle_area: float
    dry_mass: float
    wet_mass: float
    vac_flag: float
    propellant_mass: float
    burn_time: float
    mass_dot: DataTable
    cg: DataTable
    thrust: DataTable

    table_names = ("mass_dot",
                   "cg",
                   "thrust")

    scalar_names = ("nozzle_area",
                    "dry_mass",
                    "wet_mass",
                    "vac_flag",
                    "propellant_mass",
                    "burn_time")

    cpp: CppMassTable

    def __new__(cls, path: str|Path = "",
                keep_units: bool = False,
                angle_units: Unit = u.rad,  # type:ignore
                length_units: Unit = u.m,  # type:ignore
                mass_units: Unit = u.kg,  # type:ignore
                force_units: Unit = u.N,  # type:ignore
                **kwargs,
                ):
        obj = super().__new__(cls)

        if path:
            matfile = MatFile(path)
            obj._build_from_matfile(matfile,
                                    keep_units=keep_units,
                                    angle_units=angle_units,
                                    length_units=length_units,
                                    mass_units=mass_units,
                                    force_units=force_units)
            cpp_tables = obj._get_cpp_tables()
            scalars = obj._get_scalars()
            obj.cpp = CppMassTable(**cpp_tables, **scalars)
        else:
            # handle tables passed as kwargs
            cpp_table_inits = {}  # table inits dictionary for cpp-init
            for name in obj.table_names:
                if name in kwargs:
                    table = kwargs.get(name)
                    setattr(obj, name, table)
                    cpp_table_inits[name] = table.cpp  # type:ignore
                else:
                    # -----------------------------------------------------
                    # TODO: right now cpp_default is different than
                    # py_default. cpp_datatable cannot accept None as
                    # data or empty dict as axes. need to make this
                    # consitent. py-side AeroTable reliles on use of
                    # isnone() method (which relies on DataTable(data=None))
                    # -----------------------------------------------------
                    py_default = DataTable(None, {})
                    cpp_default = DataTable(np.array([]), {"null": np.array([])})
                    # cpp_default = DataTable([], {})
                    setattr(obj, name, py_default)
                    cpp_table_inits[name] = cpp_default.cpp

            # handle scalar values passed as kwargs
            cpp_scalar_inits = {}
            for name in obj.scalar_names:
                if name in kwargs:
                    scalar = kwargs.get(name)
                    setattr(obj, name, scalar)
                    cpp_scalar_inits[name] = scalar

            obj.cpp = CppMassTable(**cpp_table_inits, **cpp_scalar_inits)

        return obj


    def _build_from_matfile(self, file: MatFile,
                            keep_units: bool = False,
                            angle_units: Unit = u.rad,  # type:ignore
                            length_units: Unit = u.m,  # type:ignore
                            mass_units: Optional[Unit] = None,  # type:ignore
                            force_units: Optional[Unit] = None,  # type:ignore
                            ):
        """Attempts to auto-build MassTable from provided MatFile.
        makes assumptions about each DataTables' axes / axes order.
        Unless specific unit arguments are provided, SI units will
        be assumed.

        -------------------------------------------------------------------

        Parameters:
            file: a MatFile object

            angle_units: astropy unit for all angles

            length_units: astropy unit for all lengths

        Returns:
            an MassTable

        -------------------------------------------------------------------
        """
        # NOTE: this will search the MatFile for any attributes unit related
        unit_info = file.findall("*unit*", case_sensitive=False)

        # -----------------------------------------------------------------
        # handle units & conversion behavior:
        # -----------------------------------------------------------------
        if keep_units:
            angle_conv_const = 1.0
            length_conv_const = 1.0
            area_conv_const = 1.0
            mass_conv_const = 1.0
            force_conv_const = 1.0
        else:
            angle_conv_const = float((1.0 * angle_units).si.to_value())  # type:ignore
            length_conv_const = float((1.0 * length_units).si.to_value())  # type:ignore
            area_conv_const = float((1.0 * length_units**2).si.to_value())  # type:ignore
            if mass_units:
                mass_conv_const = float((1.0 * mass_units).si.to_value())  # type:ignore
            else:
                mass_conv_const = 1.0
            if force_units:
                force_conv_const = float((1.0 * force_units).si.to_value())  # type:ignore
            else:
                force_conv_const = 1.0

        mass_dot = file.find(["mass_dot", "MassDot", "Mdot"])
        cg = file.find(["cg", "CenterOfGravity", "center_of_gravity"])
        thrust = file.find(["thrust"])
        propellant_mass = file.find(["prop_mass", "PropMass", "propellant_mass"])
        burn_time = file.find(["burn_time", "BurnTime", "StageTime"])

        mass_dot = mass_dot * mass_conv_const
        cg = cg * length_conv_const
        thrust = thrust * force_conv_const

        # special processing for `cg`:
        # if nan's are found in array, fill will last
        # known number in array indices
        nan_ids = np.where(np.isnan(cg))[0]
        if len(nan_ids):
            last_number_id = nan_ids[0] - 1
            cg[nan_ids] = cg[last_number_id]

        # -----------------------------------------------------------------
        # convert to DataTables
        # -----------------------------------------------------------------
        time_axis = {"t": burn_time}
        self.mass_dot = DataTable(mass_dot, time_axis)
        self.cg = DataTable(cg, time_axis)
        self.thrust = DataTable(thrust, time_axis)

        self.burn_time = burn_time
        self.burn_time_max = max(burn_time)
        self.propellant_mass = propellant_mass * mass_conv_const

        # scalar members
        self.nozzle_area = file.find(["nozzle_area", "NozzleArea"])
        self.dry_mass = file.find(["dry_mass", "DryMass"])
        self.wet_mass = file.find(["wet_mass", "WetMass"])
        self.vac_flag = file.find(["vac_flag", "VacFlag"])

        self.nozzle_area *= area_conv_const
        self.dry_mass *= mass_conv_const
        self.wet_mass *= mass_conv_const

        # -----------------------------------------------------------------
        # calculations for burn_ratio
        # NOTE: this is not currently used
        # -----------------------------------------------------------------
        self.burn_ratio = file.find(["burn_ratio", "BurnRatio"], default=None)
        # calculate burn ratio if not already defined
        if self.burn_ratio is None:
            # wet_mass = data.get("wet_mass")
            # dry_mass = data.get("dry_mass")
            # prop_mass = data.get("prop_mass")
            if (self.wet_mass - self.dry_mass) == 0.0:
                self.burn_ratio = np.zeros_like(self.wet_mass)
            else:
                self.burn_ratio = (
                        (np.array([self.wet_mass] * len(self.propellant_mass)) - self.propellant_mass
                         - np.array([self.dry_mass] * len(self.propellant_mass)))
                        / (self.wet_mass - self.dry_mass))


    def add_stage(self, child) -> None:
        """Adds a child AeroTable object as an ordered child
        of this object."""
        # NOTE: overloaded for the purpose of calling `cpp` member
        super().add_stage(child)
        self.cpp.add_stage(child.cpp)


    def set_stage(self, stage: int) -> None:
        """Set the current stage index for the aerotable. This is
        so that `get_stage()` will return the corresponding aerotable."""
        # NOTE: overloaded for the purpose of calling `cpp` member
        super().set_stage(stage)
        self.cpp.set_stage(stage)


    def _get_scalars(self) -> dict:
        """Returns dict of registered scalar values."""
        cpp_scalars = {}
        for name in self.scalar_names:
            if hasattr(self, name):
                if (scalar := getattr(self, name)) is not None:
                    cpp_scalars[name] = scalar
        return cpp_scalars


    def _get_cpp_tables(self) -> dict:
        """Returns dict of active cpp tables. Active cpp tables
        are tables which have been initialized and so DataTable.cpp
        is also initialized. The `cpp` attribute of DataTable is
        the c++ class found in `include/datatable.cpp`."""
        cpp_tables = {}
        for name in self.table_names:
            if hasattr(self, name):
                if not (table := getattr(self, name)).isnone():
                    cpp_tables[name] = table.cpp
        return cpp_tables


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
            return stage.mass_dot(t=t)  # type:ignore


    def get_cg(self, t: float) -> float:
        stage = self.get_stage()
        if t >= stage.burn_time_max:
            return stage.cg(t=stage.burn_time_max)  # type:ignore
        else:
            return stage.cg(t=t)  # type:ignore


    def get_isp(self, t: float, pressure: float) -> float:
        stage = self.get_stage()
        thrust = stage.get_thrust(t=t, pressure=pressure)  # type:ignore
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
            return stage.thrust(t=t)  # type:ignore


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
