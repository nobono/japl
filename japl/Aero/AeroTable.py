from typing import Optional
import numpy as np
import dill as pickle
from astropy import units as u
from japl.Util.Matlab import MatFile
from japl.Util.Util import flatten_list
from japl.DataTable.DataTable import DataTable
from japl.DataTable.DataTable import ArgType
from scipy.optimize import minimize_scalar



class Increments:
    alpha = np.empty([])
    phi = np.empty([])
    mach = np.empty([])
    alt = np.empty([])
    iota = np.empty([])
    iota_prime = np.empty([])

    nalpha = 0
    nphi = 0
    nmach = 0
    nalt = 0
    niota = 0
    niota_prime = 0


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


def swap_to_correct_axes(array: np.ndarray, labels: list[str]) -> np.ndarray:
    default_label_order = ["alpha", "phi", "mach", "alt", "iota"]
    id_swap_order = []
    for label in default_label_order:
        if label in labels:
            id_swap_order += [labels.index(label)]
    return np.transpose(array, id_swap_order)


# TODO: do this better?
def from_CMS_table(data: MatFile|dict, units: str = "si") -> tuple[MatFile|dict, tuple]:
    # ft2m = (1.0 * u.imperial.foot).to_value(u.m)  # type:ignore
    deg2rad = (np.pi / 180.0)

    convert_key_map = [("Alpha", "alpha"),
                       ("Mach", "mach"),
                       ("Alt", "alt"),
                       ("CA_Boost", "CA_Powered"),
                       ("CA_Coast", "CA_Unpowered"),
                       ("CNB", "CN")]

    # store to correct attribute name
    for map in convert_key_map:
        key_out, key_in = map
        table: np.ndarray = data.get(key_in)  # type:ignore
        if "CA" in key_in or "CN" in key_in:
            # swap to correct axis labels
            axis_labels = flatten_list(data.get(f"{key_in}_GridLabels"))  # type:ignore
            table = swap_to_correct_axes(table, axis_labels)
        setattr(data, key_out, table)
        delattr(data, key_in)

    alpha = data.get("Alpha", None)
    mach = data.get("Mach", None)
    alt = data.get("Alt", None)

    ############################################################
    # Convert to SI units
    # NOTE: CMS stores aerodata length units as meters so
    # no SI conversion needed
    ############################################################

    if units.lower() == "si":
        alpha = alpha * deg2rad

    alpha = alpha.astype(np.float64)
    mach = mach.astype(np.float64)
    alt = alt.astype(np.float64)

    # table shape axes
    Basic_axes = {"alpha": alpha, "mach": mach}                 # Basic table shape
    CA_axes = {"mach": mach, "alt": alt}                        # CA-coeff table shape
    CNB_axes = {"alpha": alpha, "mach": mach}
    IT_axes = {"alpha": alpha, "mach": mach}                    # fin-increment table shape
    CA_Total_axes = {"alpha": alpha, "mach": mach, "alt": alt}  # CA-coeff total table shape

    table_axes = (Basic_axes, CA_axes, CNB_axes, IT_axes, CA_Total_axes)
    return (data, table_axes)


def from_orion_table(data: MatFile|dict, units: str = "si") -> tuple[MatFile|dict, tuple]:
    deg2rad = np.radians(1)

    data, table_axes = from_CMS_table(data=data, units=units)
    (Basic_axes,
     CA_axes,
     CNB_axes,
     IT_axes,
     CA_Total_axes) = table_axes

    alpha = data.get("Alpha", None)
    mach = data.get("Mach", None)
    alt = data.get("Alt", None)

    if units.lower() == "si":
        alpha = alpha * deg2rad

    alpha = alpha.astype(np.float64)
    mach = mach.astype(np.float64)
    alt = alt.astype(np.float64)

    # adding alt in CNB table
    CNB_axes = {"alpha": alpha, "mach": mach, "alt": alt}
    table_axes = (Basic_axes, CA_axes, CNB_axes, IT_axes, CA_Total_axes)
    return (data, table_axes)


def from_default_table(data: MatFile|dict, units: str = "si") -> tuple[MatFile|dict, tuple]:
    ft2m = (1.0 * u.imperial.foot).to_value(u.m)  # type:ignore
    inch2m = (1.0 * u.imperial.inch).to_value(u.m)  # type:ignore
    deg2rad = (np.pi / 180.0)
    # lbminch2Nm = (1.0 * u.imperial.lbm * u.imperial.inch**2).to_value(u.kg * u.m**2)  # type:ignore
    inch_sq_2_m_sq = (1.0 * u.imperial.inch**2).to_value(u.m**2)  # type:ignore

    alpha = data.get("Alpha", None)
    phi = data.get("Phi", None)
    mach = data.get("Mach", None)
    alt = data.get("Alt", None)
    iota = data.get("Iota", None)
    # iota_prime = data.get("Iota_Prime", None)

    ############################################################
    # Convert to SI units
    # TODO make input and ouput of units better...
    # NOTE: currently aero data available is in imperial units
    ############################################################
    if "Sref" in data:
        Sref = getattr(data, "Sref")
        setattr(data, "Sref", Sref * inch_sq_2_m_sq)
    else:
        raise Exception("aerodata has no \"Sref\" attribute")
    if "Lref" in data:
        Sref = getattr(data, "Lref")
        setattr(data, "Lref", Sref * inch2m)
    else:
        raise Exception("aerodata has no \"Lref\" attribute")
    if "MRC" in data:
        Sref = getattr(data, "MRC")
        setattr(data, "MRC", Sref * inch2m)
    else:
        raise Exception("aerodata has no \"MRC\" (Missile Reference Center) attribute")

    if units.lower() == "si":
        alpha = alpha * deg2rad
        phi = phi * deg2rad
        alt = alt * ft2m
        iota = iota * deg2rad

    alpha = alpha.astype(np.float64)
    phi = phi.astype(np.float64)
    mach = mach.astype(np.float64)
    alt = alt.astype(np.float64)
    iota = iota.astype(np.float64)

    Basic_axes = {"alpha": alpha, "phi": phi, "mach": mach}                               # Basic table shape
    CA_axes = {"phi": phi, "mach": mach, "alt": alt}                                      # CA-coeff table shape
    CNB_axes = {"alpha": alpha, "phi": phi, "mach": mach, "iota": iota}                   # Basic table shape
    IT_axes = {"alpha": alpha, "phi": phi, "mach": mach, "iota": iota}                    # fin-increment table shape
    CA_Total_axes = {"alpha": alpha, "phi": phi, "mach": mach, "alt": alt, "iota": iota}  # CA-coeff total table shape

    table_axes = (Basic_axes, CA_axes, CNB_axes, IT_axes, CA_Total_axes)
    return (data, table_axes)


class AeroTable:

    """This class is for containing Aerotable data for a particular
    SimObject."""

    # __ft2m = (1.0 * u.imperial.foot).to_value(u.m)  # type:ignore
    # __inch2m = (1.0 * u.imperial.inch).to_value(u.m)  # type:ignore
    # __deg2rad = (np.pi / 180.0)
    # __lbminch2Nm = (1.0 * u.imperial.lbm * u.imperial.inch**2).to_value(u.kg * u.m**2)  # type:ignore
    # __inch_sq_2_m_sq = (1.0 * u.imperial.inch**2).to_value(u.m**2)  # type:ignore

    def __init__(self, data: Optional[str|dict|MatFile] = None, from_template: str = "", units: str = "si") -> None:
        self.units = units

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
            case "orion":
                data_dict, table_axes = from_orion_table(data_dict, units=units)
            case _:
                data_dict, table_axes = from_default_table(data_dict, units=units)

        ############################################################
        # Load tables from data
        ############################################################
        _CA_inv = data_dict.get("CA_inv", None)          # (alpha, phi, mach)
        _CA_Basic = data_dict.get("CA_Basic", None)      # (alpha, phi, mach)
        _CA_0_Boost = data_dict.get("CA_0_Boost", None)  # (phi, mach, alt)
        _CA_0_Coast = data_dict.get("CA_0_Coast", None)  # (phi, mach, alt)
        _CA_IT = data_dict.get("CA_IT", None)            # (alpha, phi, mach, iota)
        _CYB_Basic = data_dict.get("CYB_Basic", None)    # (alpha, phi, mach)
        _CYB_IT = data_dict.get("CYB_IT", None)          # (alpha, phi, mach, iota)
        _CNB_Basic = data_dict.get("CNB_Basic", None)    # (alpha, phi, mach)
        _CNB_IT = data_dict.get("CNB_IT", None)          # (alpha, phi, mach, iota)
        _CLLB_Basic = data_dict.get("CLLB_Basic", None)  # (alpha, phi, mach)
        _CLLB_IT = data_dict.get("CLLB_IT", None)        # (alpha, phi, mach, iota)
        _CLMB_Basic = data_dict.get("CLMB_Basic", None)  # (alpha, phi, mach)
        _CLMB_IT = data_dict.get("CLMB_IT", None)        # (alpha, phi, mach, iota)
        _CLNB_Basic = data_dict.get("CLNB_Basic", None)  # (alpha, phi, mach)
        _CLNB_IT = data_dict.get("CLNB_IT", None)        # (alpha, phi, mach, iota)
        _Fin2_CN = data_dict.get("Fin2_CN", None)        # (alpha, phi, mach, iota)
        _Fin2_CBM = data_dict.get("Fin2_CBM", None)      # (alpha, phi, mach, iota)
        _Fin2_CHM = data_dict.get("Fin2_CHM", None)      # (alpha, phi, mach, iota)
        _Fin4_CN = data_dict.get("Fin4_CN", None)        # (alpha, phi, mach, iota)
        _Fin4_CBM = data_dict.get("Fin4_CBM", None)      # (alpha, phi, mach, iota)
        _Fin4_CHM = data_dict.get("Fin4_CHM", None)      # (alpha, phi, mach, iota)

        _CA_Boost = data_dict.get("CA_Boost", None)  # (alpha, phi, mach, alt, iota)
        _CA_Coast = data_dict.get("CA_Coast", None)  # (alpha, phi, mach, alt, iota)
        _CNB = data_dict.get("CNB", None)            # (alpha, phi, mach, iota)
        _CLMB = data_dict.get("CLMB", None)          # (alpha, phi, mach, iota)
        _CLNB = data_dict.get("CLNB", None)          # (alpha, phi, mach, iota)
        _CYB = data_dict.get("CYB", None)            # (alpha, phi, mach, iota)

        self.Sref = data_dict.get("Sref", 0.0)  # surface area reference
        self.Lref = data_dict.get("Lref", 0.0)  # length reference
        self.MRC = data_dict.get("MRC", 0.0)    # missile reference center

        ############################################################
        # Initialize as DataTables
        ############################################################
        # establish default axes
        #
        # - default tables axes are the expected axes for each type
        #   type of table.
        #
        # - four types of tables axes:
        #       Basic: (alpha, phi, mach)
        #       CA: (phi, mach, alt)
        #       IT: fin-increments over (phi, mach, alt)
        #       Total: combined tables "Basic + IT + CA" over (phi, mach, alt)
        #
        # - this helps to initialize DataTables dynamically from various input
        #   sources; where some axes (i.e.) "alt" may or may not be present.
        #
        ############################################################

        (Basic_axes,
         CA_axes,
         CNB_axes,
         IT_axes,
         CA_Total_axes) = table_axes

        # store increments for ease of access
        _total_axes = {}
        _total_axes.update(Basic_axes)
        _total_axes.update(CA_axes)
        _total_axes.update(IT_axes)
        _total_axes.update(CA_Total_axes)
        self.increments = Increments()
        if "alpha" in _total_axes:
            self.increments.alpha = _total_axes["alpha"]
        if "phi" in _total_axes:
            self.increments.phi = _total_axes["phi"]
        if "mach" in _total_axes:
            self.increments.mach = _total_axes["mach"]
        if "alt" in _total_axes:
            self.increments.alt = _total_axes["alt"]
        if "iota" in _total_axes:
            self.increments.iota = _total_axes["iota"]

        # create DataTables
        self.CA_inv = DataTable(_CA_inv, Basic_axes)
        self.CA_Basic = DataTable(_CA_Basic, Basic_axes)
        self.CA_0_Boost = DataTable(_CA_0_Boost, CA_axes)
        self.CA_0_Coast = DataTable(_CA_0_Coast, CA_axes)
        self.CA_IT = DataTable(_CA_IT, IT_axes)
        self.CYB_Basic = DataTable(_CYB_Basic, Basic_axes)
        self.CYB_IT = DataTable(_CYB_IT, IT_axes)
        self.CNB_Basic = DataTable(_CNB_Basic, Basic_axes)
        self.CNB_IT = DataTable(_CNB_IT, IT_axes)
        self.CLLB_Basic = DataTable(_CLLB_Basic, Basic_axes)
        self.CLLB_IT = DataTable(_CLLB_IT, IT_axes)
        self.CLMB_Basic = DataTable(_CLMB_Basic, Basic_axes)
        self.CLMB_IT = DataTable(_CLMB_IT, IT_axes)
        self.CLNB_Basic = DataTable(_CLNB_Basic, Basic_axes)
        self.CLNB_IT = DataTable(_CLNB_IT, IT_axes)
        self.Fin2_CN = DataTable(_Fin2_CN, IT_axes)
        self.Fin2_CBM = DataTable(_Fin2_CBM, IT_axes)
        self.Fin2_CHM = DataTable(_Fin2_CHM, IT_axes)
        self.Fin4_CN = DataTable(_Fin4_CN, IT_axes)
        self.Fin4_CBM = DataTable(_Fin4_CBM, IT_axes)
        self.Fin4_CHM = DataTable(_Fin4_CHM, IT_axes)

        self.CA_Boost = DataTable(_CA_Boost, CA_Total_axes)
        self.CA_Coast = DataTable(_CA_Coast, CA_Total_axes)
        self.CNB = DataTable(_CNB, CNB_axes)
        self.CLMB = DataTable(_CLMB, IT_axes)
        self.CLNB = DataTable(_CLNB, IT_axes)
        self.CYB = DataTable(_CYB, IT_axes)

        # MRC may be a float or array
        # TODO: maybe do this better
        if hasattr(self.MRC, "__len__"):
            self.MRC = np.asarray(self.MRC, dtype=float)
        else:
            self.MRC = float(self.MRC)

        ############################################################
        # Build Total DataTables from sub-tables
        #   (Basic + Increment) tables
        ############################################################
        if self.CA_Boost.isnone():
            self.CA_Boost = DataTable(
                    self.CA_Basic[:, :, :, np.newaxis, np.newaxis]
                    + self.CA_0_Boost[np.newaxis, :, :, :, np.newaxis]
                    + self.CA_IT[:, :, :, np.newaxis, :],
                    axes=CA_Total_axes)
        if self.CA_Coast.isnone():
            self.CA_Coast = DataTable(
                    self.CA_Basic[:, :, :, np.newaxis, np.newaxis]
                    + self.CA_0_Coast[np.newaxis, :, :, :, np.newaxis]
                    + self.CA_IT[:, :, :, np.newaxis, :],
                    axes=CA_Total_axes)
        if self.CNB.isnone():
            if not (self.CNB_Basic.isnone() and self.CNB_IT.isnone()):
                self.CNB = DataTable(self.CNB_Basic[:, :, :, np.newaxis]
                                     + self.CNB_IT,
                                     axes=CNB_axes)
        if self.CLMB.isnone():
            if not (self.CLMB_Basic.isnone() and self.CLMB_IT.isnone()):
                self.CLMB = DataTable(self.CLMB_Basic[:, :, :, np.newaxis]
                                      + self.CLMB_IT,
                                      axes=IT_axes)
        if self.CLNB.isnone():
            if not (self.CLNB_Basic.isnone() and self.CLNB_IT.isnone()):
                self.CLNB = DataTable(self.CLNB_Basic[:, :, :, np.newaxis]
                                      + self.CLNB_IT,
                                      axes=IT_axes)
        if self.CYB.isnone():
            if not (self.CYB_Basic.isnone() and self.CYB_IT.isnone()):
                self.CYB = DataTable(self.CYB_Basic[:, :, :, np.newaxis]
                                     + self.CYB_IT,
                                     axes=IT_axes)

        ############################################################
        # For Momementless Dynamics, the following tables are
        #   - compute CA_Boost wrt. alpha
        #   - compute CA_Boost wrt. alpha
        #   - compute CNB wrt. alpha
        ############################################################
        if units.lower() == "si":
            delta_alpha = np.radians(0.1)
        else:
            delta_alpha = 0.1
        self.CA_Boost_alpha = self.create_diff_table(table=self.CA_Boost,
                                                     diff_arg="alpha",
                                                     delta_arg=delta_alpha)

        self.CA_Coast_alpha = self.create_diff_table(table=self.CA_Coast,
                                                     diff_arg="alpha",
                                                     delta_arg=delta_alpha)

        self.CNB_alpha = self.create_diff_table(table=self.CNB,
                                                diff_arg="alpha",
                                                delta_arg=delta_alpha)

        ############################################################
        # Excpected DataTable names
        ############################################################
        CA_0_table_names = ["CA_0_Boost",
                            "CA_0_Coast"]
        Basic_table_names = ["CA_inv",
                             "CA_Basic",
                             "CYB_Basic",
                             "CNB_Basic",
                             "CLLB_Basic",
                             "CLMB_Basic",
                             "CLNB_Basic"]
        Iter_table_names = ["CA_IT",
                            "CYB_IT",
                            "CNB_IT",
                            "CLLB_IT",
                            "CLMB_IT",
                            "CLNB_IT",
                            "Fin2_CN",
                            "Fin2_CBM",
                            "Fin2_CHM",
                            "Fin4_CN",
                            "Fin4_CBM",
                            "Fin4_CHM"]
        Total_table_names = ["CA_Boost",
                             "CA_Coast",
                             "CNB",
                             "CLMB",
                             "CLNB",
                             "CYB"]
        Diff_table_names = ["CA_Boost_alpha",
                            "CA_Coast_alpha",
                            "CNB_alpha"]
        table_names = (CA_0_table_names + Basic_table_names + Iter_table_names
                       + Total_table_names + Diff_table_names)

        ############################################################
        # Table Reflections
        #   if alpha is not reflected across 0, so mirror the tables
        #   across alpha increments
        ############################################################
        # check if alpha increments are reflected across zero
        # or need to be mirrored
        if Basic_axes["alpha"][0] == 0:
            alpha_axis = 0
            for table_name in table_names:
                _table = getattr(self, table_name)
                if not _table.isnone():
                    if "alpha" in _table.axes:
                        mirrored_table = _table.mirror_axis(alpha_axis)
                        setattr(self, table_name, mirrored_table)

        # modules for symbolic mapping
        self.modules = {
                "aerotable_increments": self.increments,
                "aerotable_get_CA": self.get_CA,
                "aerotable_get_CA_Boost": self.get_CA_Boost,
                "aerotable_get_CA_Coast": self.get_CA_Coast,
                "aerotable_get_CNB": self.get_CNB,
                "aerotable_get_CLMB": self.get_CLMB,
                "aerotable_get_CLNB": self.get_CLNB,
                "aerotable_get_CYB": self.get_CYB,
                "aerotable_get_MRC": self.get_MRC,
                "aerotable_get_Sref": self.get_Sref,
                "aerotable_get_Lref": self.get_Lref,
                "aerotable_get_CA_Boost_alpha": self.get_CA_Boost_alpha,
                "aerotable_get_CA_Coast_alpha": self.get_CA_Coast_alpha,
                "aerotable_get_CNB_alpha": self.get_CNB_alpha,
                "aerotable_inv_aerodynamics": self.inv_aerodynamics,
                }

    ############################################################
    # Methods
    ############################################################


    @staticmethod
    def create_mirrored_array(array: np.ndarray) -> np.ndarray:
        return np.concatenate([-array[::-1][:-1], array])


    def create_diff_table(self, table: DataTable,
                          diff_arg: str, delta_arg: float) -> DataTable:
        """This method differentiates a table wrt. an increment variable
        name \"diff_arg\"."""
        # get min and max values to keep diff within table range
        max_alpha = max(table.axes.get(diff_arg))  # type:ignore
        min_alpha = min(table.axes.get(diff_arg))  # type:ignore

        # handle table args
        val_args = self._get_table_args(table, **table.axes)
        arg_grid = np.meshgrid(*val_args, indexing="ij")
        args = {str(k): v for k, v in zip(table.axes, arg_grid)}

        # create diff_arg plus & minus values for linear interpolation
        args_plus = args.copy()
        args_minus = args.copy()
        args_plus[diff_arg] = np.clip(args[diff_arg] + delta_arg, min_alpha, max_alpha)
        args_minus[diff_arg] = np.clip(args[diff_arg] - delta_arg, min_alpha, max_alpha)

        val_plus = table(**args_plus)
        val_minus = table(**args_minus)
        diff_table = (val_plus - val_minus) / (args_plus[diff_arg] - args_minus[diff_arg])  # type:ignore
        return DataTable(diff_table, axes=table.axes)


    # def _get_CA_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach),
    #                    self.CA_Basic,
    #                    [abs(alpha), phi, mach],
    #                    method=method)[0]

    # def _get_CA_0_Boost(self, phi: float, mach: float, alt: float, method: str = "linear") -> float:
    #     return interpn((self.increments.phi, self.increments.mach, self.increments.alt),
    #                    self.CA_0_Boost,
    #                    [phi, mach, alt],
    #                    method=method)[0]


    # def _get_CA_0_Coast(self, phi: float, mach: float, alt: float, method: str = "linear") -> float:
    #     return interpn((self.increments.phi, self.increments.mach, self.increments.alt),
    #                    self.CA_0_Coast,
    #                    [phi, mach, alt],
    #                    method=method)[0]


    # def _get_CA_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota),
    #                    self.CA_IT,
    #                    [abs(alpha), phi, mach, iota],
    #                    method=method)[0]


    # def _get_CNB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach),
    #                    self.CNB_Basic,
    #                    [alpha, phi, mach],
    #                    method=method)[0]


    # def _get_CNB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota),
    #                    self.CNB_IT,
    #                    [alpha, phi, mach, iota],
    #                    method=method)[0]


    # def _get_CYB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach),
    #                    self.CYB_Basic,
    #                    [abs(alpha), phi, mach],
    #                    method=method)[0]


    # def _get_CYB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota),
    #                    self.CYB_IT,
    #                    [abs(alpha), phi, mach, iota],
    #                    method=method)[0]


    # def _get_CLMB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach),
    #                    self.CLMB_Basic,
    #                    [alpha, phi, mach],
    #                    method=method)[0]


    # def _get_CLMB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota),
    #                    self.CLMB_IT,
    #                    [alpha, phi, mach, iota],
    #                    method=method)[0]


    # def _get_CLNB_Basic(self, alpha: float, phi: float, mach: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach),
    #                    self.CLNB_Basic,
    #                    [abs(alpha), phi, mach],
    #                    method=method)[0]


    # def _get_CLNB_IT(self, alpha: float, phi: float, mach: float, iota: float, method: str = "linear") -> float:
    #     return interpn((self.increments.alpha, self.increments.phi, self.increments.mach, self.increments.iota),
    #                    self.CLNB_IT,
    #                    [abs(alpha), phi, mach, iota],
    #                    method=method)[0]


    def set(self, aerotable: "AeroTable") -> None:
        """This method re-initializes the aerotables with the
        provided AeroTable argument. This is to provide different
        tables if a model switches stages."""
        table_names = ["CA_0_Boost",
                       "CA_0_Coast",
                       "CA_inv",
                       "CA_Basic",
                       "CYB_Basic",
                       "CNB_Basic",
                       "CLLB_Basic",
                       "CLMB_Basic",
                       "CLNB_Basic",
                       "CA_IT",
                       "CYB_IT",
                       "CNB_IT",
                       "CLLB_IT",
                       "CLMB_IT",
                       "CLNB_IT",
                       "Fin2_CN",
                       "Fin2_CBM",
                       "Fin2_CHM",
                       "Fin4_CN",
                       "Fin4_CBM",
                       "Fin4_CHM",
                       "CA_Boost",
                       "CA_Coast",
                       "CNB",
                       "CLMB",
                       "CLNB",
                       "CYB",
                       "CA_Boost_alpha",
                       "CA_Coast_alpha",
                       "CNB_alpha"]
        for name in table_names:
            setattr(self, name, getattr(aerotable, name))


    def ld_guidance(self,
                    alpha: float,
                    phi: Optional[float] = None,
                    mach: Optional[float] = None,
                    alt: Optional[float] = None,
                    iota: Optional[float] = None):
        alpha_max = self.CNB.axes["alpha"].max()
        alpha_min = self.CNB.axes["alpha"].min()
        # if alpha > alpha_max or alpha < alpha_min:
        #     print("alpha clipping....")

        def ld_ratio(_alpha):
            CA = self.get_CA(alpha=_alpha, phi=phi, mach=mach, alt=alt, iota=iota)
            CN = self.get_CNB(alpha=_alpha, phi=phi, mach=mach, alt=alt, iota=iota)
            cosa = np.cos(_alpha)
            sina = np.sin(_alpha)
            CL = (CN * cosa) - (CA * sina)
            CD = (CN * sina) + (CA * cosa)
            ratio = -CL / CD
            return ratio
        result = minimize_scalar(ld_ratio, bounds=(alpha_min, alpha_max), method='bounded')
        optimal_alpha = result.x
        ld_max = -result.fun
        # optimal CL, CD
        CA = self.get_CA(alpha=optimal_alpha, phi=phi, mach=mach, alt=alt, iota=iota)
        CN = self.get_CNB(alpha=optimal_alpha, phi=phi, mach=mach, alt=alt, iota=iota)
        cosa = np.cos(optimal_alpha)
        sina = np.sin(optimal_alpha)
        opt_CL = (CN * cosa) - (CA * sina)
        opt_CD = (CN * sina) + (CA * cosa)

        optimal_alpha = np.clip(optimal_alpha, alpha_min, alpha_max)
        return opt_CL, opt_CD, optimal_alpha


    def inv_aerodynamics(self,
                         thrust: float,
                         acc_cmd: float,
                         dynamic_pressure: float,
                         mass: float,
                         alpha: float,
                         phi: Optional[float] = None,
                         mach: Optional[float] = None,
                         alt: Optional[float] = None,
                         iota: Optional[float] = None) -> float:
        if self.units.lower() == "si":
            alpha_tol = np.radians(0.01)
        else:
            alpha_tol = 0.01

        alpha_max = max(self.increments.alpha)
        Sref = self.get_Sref()

        alpha_last = -1000
        count = 0

        # gradient search
        while ((abs(alpha - alpha_last) > alpha_tol) and (count < 10)):  # type:ignore
            count += 1
            alpha_last = alpha

            # TODO switch between Boost / Coast
            # get coeffs from aerotable
            CA = self.get_CA(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)
            CN = self.get_CNB(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)
            CA_alpha = self.get_CA_alpha(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)
            CN_alpha = self.get_CNB_alpha(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)

            # get derivative of CL wrt alpha
            cosa = np.cos(alpha)
            sina = np.sin(alpha)
            CL = (CN * cosa) - (CA * sina)
            CL_alpha = ((CN_alpha - CA) * cosa) - ((CA_alpha + CN) * sina)
            # CD = (CN * sina) + (CA * cosa)
            # CD_alpha = ((CA_alpha + CN) * cosa) + ((CN_alpha - CA) * sina)

            # calculate current normal acceleration, acc0, and normal acceleration due to
            # the change in alpha, acc_alpha. Use the difference between the two to
            # iteratively update alpha.
            acc_alpha = CL_alpha * dynamic_pressure * Sref / mass + thrust * np.cos(alpha) / mass
            acc0 = CL * dynamic_pressure * Sref / mass + thrust * np.sin(alpha) / mass
            alpha = alpha + (acc_cmd - acc0) / acc_alpha
            alpha = max(0, min(alpha, alpha_max))

        angle_of_attack = alpha
        return angle_of_attack


    def get_CA(self,
               boosting: bool = True,
               alpha: Optional[ArgType] = None,
               phi: Optional[ArgType] = None,
               mach: Optional[ArgType] = None,
               alt: Optional[ArgType] = None,
               iota: Optional[ArgType] = None) -> float|np.ndarray:
        if boosting:
            return self.get_CA_Boost(abs(alpha), phi, mach, alt, iota)  # type:ignore
        else:
            return self.get_CA_Coast(abs(alpha), phi, mach, alt, iota)  # type:ignore


    def get_CA_alpha(self,
                     boosting: bool = True,
                     alpha: Optional[ArgType] = None,
                     phi: Optional[ArgType] = None,
                     mach: Optional[ArgType] = None,
                     alt: Optional[ArgType] = None,
                     iota: Optional[ArgType] = None) -> float|np.ndarray:
        if boosting:
            return self.get_CA_Boost_alpha(abs(alpha), phi, mach, alt, iota)  # type:ignore
        else:
            return self.get_CA_Coast_alpha(abs(alpha), phi, mach, alt, iota)  # type:ignore


    def get_CA_Boost(self,
                     alpha: Optional[ArgType] = None,
                     phi: Optional[ArgType] = None,
                     mach: Optional[ArgType] = None,
                     alt: Optional[ArgType] = None,
                     iota: Optional[ArgType] = None) -> float|np.ndarray:
        # TODO do this better
        # protection / boundary for alpha
        if alpha is not None:
            alpha = np.clip(alpha, self.CA_Boost.axes["alpha"].min(), self.CA_Boost.axes["alpha"].max())
        return self.CA_Boost(alpha=abs(alpha), phi=phi, mach=mach, alt=alt, iota=iota)  # type:ignore


    def get_CA_Coast(self,
                     alpha: Optional[ArgType] = None,
                     phi: Optional[ArgType] = None,
                     mach: Optional[ArgType] = None,
                     alt: Optional[ArgType] = None,
                     iota: Optional[ArgType] = None) -> float|np.ndarray:
        if alpha is not None:
            alpha = np.clip(alpha, self.CA_Boost.axes["alpha"].min(), self.CA_Boost.axes["alpha"].max())
        return self.CA_Coast(alpha=abs(alpha), phi=phi, mach=mach, alt=alt, iota=iota)  # type:ignore


    def get_CNB(self,
                alpha: Optional[ArgType] = None,
                phi: Optional[ArgType] = None,
                mach: Optional[ArgType] = None,
                alt: Optional[ArgType] = None,
                iota: Optional[ArgType] = None) -> float|np.ndarray:
        if alpha is not None:
            alpha = np.clip(alpha, self.CA_Boost.axes["alpha"].min(), self.CA_Boost.axes["alpha"].max())
        return self.CNB(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)


    def get_CLMB(self,
                 alpha: Optional[ArgType] = None,
                 phi: Optional[ArgType] = None,
                 mach: Optional[ArgType] = None,
                 iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CLMB(alpha=alpha, phi=phi, mach=mach, iota=iota)


    def get_CLNB(self,
                 alpha: Optional[ArgType] = None,
                 phi: Optional[ArgType] = None,
                 mach: Optional[ArgType] = None,
                 iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CLNB(alpha=alpha, phi=phi, mach=mach, iota=iota)


    def get_CYB(self,
                alpha: Optional[ArgType] = None,
                phi: Optional[ArgType] = None,
                mach: Optional[ArgType] = None,
                iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CYB(alpha=alpha, phi=phi, mach=mach, iota=iota)


    def get_CA_Boost_alpha(self,
                           alpha: Optional[ArgType] = None,
                           phi: Optional[ArgType] = None,
                           mach: Optional[ArgType] = None,
                           alt: Optional[ArgType] = None,
                           iota: Optional[ArgType] = None) -> float|np.ndarray:
        if alpha is not None:
            alpha = np.clip(alpha, self.CA_Boost.axes["alpha"].min(), self.CA_Boost.axes["alpha"].max())
        return self.CA_Boost_alpha(alpha=abs(alpha), phi=phi, mach=mach, alt=alt, iota=iota)  # type:ignore


    def get_CA_Coast_alpha(self,
                           alpha: Optional[ArgType] = None,
                           phi: Optional[ArgType] = None,
                           mach: Optional[ArgType] = None,
                           alt: Optional[ArgType] = None,
                           iota: Optional[ArgType] = None) -> float|np.ndarray:
        if alpha is not None:
            alpha = np.clip(alpha, self.CA_Boost.axes["alpha"].min(), self.CA_Boost.axes["alpha"].max())
        return self.CA_Coast_alpha(alpha=abs(alpha), phi=phi, mach=mach, alt=alt, iota=iota)  # type:ignore


    def get_CNB_alpha(self,
                      alpha: Optional[ArgType] = None,
                      phi: Optional[ArgType] = None,
                      mach: Optional[ArgType] = None,
                      alt: Optional[ArgType] = None,
                      iota: Optional[ArgType] = None) -> float|np.ndarray:
        if alpha is not None:
            alpha = np.clip(alpha, self.CA_Boost.axes["alpha"].min(), self.CA_Boost.axes["alpha"].max())
        return self.CNB_alpha(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)


    def _get_table_args(self, table: DataTable, **kwargs) -> tuple:
        """This method handles arguments passed to DataTables dynamically
        according to the arguments passed and the axis_labels of the table
        being accessed."""
        args = ()
        for label in table.axes:
            arg_val = kwargs.get(label, None)
            if arg_val is not None:
                args += (arg_val,)
        return args


    def get_MRC(self) -> float|np.ndarray:
        if isinstance(self.MRC, np.ndarray):
            return self.MRC[0]
        else:
            return self.MRC


    def get_Sref(self) -> float:
        return self.Sref


    def get_Lref(self) -> float:
        return self.Lref


    def __repr__(self) -> str:
        members = [i for i in dir(self) if "__" not in i]
        return "\n".join(members)
