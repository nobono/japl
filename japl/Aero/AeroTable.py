from typing import Optional
import numpy as np
import pickle
from astropy import units as u
from japl.Util.Matlab import MatFile
from japl.Util.Util import flatten_list
from japl.DataTable.DataTable import DataTable
from japl.DataTable.DataTable import ArgType



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
                members += [str(i) + f" [{len(attr)}]"]
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
    __ft2m = (1.0 * u.imperial.foot).to_value(u.m)  # type:ignore
    __deg2rad = (np.pi / 180.0)

    convert_key_map = [
                       ("Alpha", "alpha"),
                       ("Mach", "mach"),
                       ("Alt", "alt"),
                       ("CA_Boost_Total", "CA_Powered"),
                       ("CA_Coast_Total", "CA_Unpowered"),
                       ("CNB_Total", "CN"),
                       ]
    # store to correct attribute name
    # CNB_labels = data.get("CN_GridLabels")
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

    if units.lower() == "si":
        alpha = alpha * __deg2rad
        alt = alt * __ft2m

    alpha = alpha.astype(np.float64)
    mach = mach.astype(np.float64)
    alt = alt.astype(np.float64)

    # table shape labels
    Basic_labels = {"alpha": alpha, "mach": mach}                 # Basic table shape
    CA_labels = {"mach": mach, "alt": alt}                        # CA-coeff table shape
    IT_labels = {"alpha": alpha, "mach": mach}                    # fin-increment table shape
    CA_Total_labels = {"alpha": alpha, "mach": mach, "alt": alt}  # CA-coeff total table shape
    table_axis_labels = (Basic_labels, CA_labels, IT_labels, CA_Total_labels)
    return (data, table_axis_labels)


def from_default_table(data: MatFile|dict, units: str = "si") -> tuple[MatFile|dict, tuple]:
    __ft2m = (1.0 * u.imperial.foot).to_value(u.m)  # type:ignore
    __deg2rad = (np.pi / 180.0)

    alpha = data.get("Alpha", None)
    phi = data.get("Phi", None)
    mach = data.get("Mach", None)
    alt = data.get("Alt", None)
    iota = data.get("Iota", None)
    # iota_prime = data.get("Iota_Prime", None)

    if units.lower() == "si":
        alpha = alpha * __deg2rad
        phi = phi * __deg2rad
        alt = alt * __ft2m
        iota = iota * __deg2rad

    alpha = alpha.astype(np.float64)
    phi = phi.astype(np.float64)
    mach = mach.astype(np.float64)
    alt = alt.astype(np.float64)
    iota = iota.astype(np.float64)

    Basic_labels = {"alpha": alpha, "phi": phi, "mach": mach}                               # Basic table shape
    CA_labels = {"phi": phi, "mach": mach, "alt": alt}                                      # CA-coeff table shape
    IT_labels = {"alpha": alpha, "phi": phi, "mach": mach, "iota": iota}                    # fin-increment table shape
    CA_Total_labels = {"alpha": alpha, "phi": phi, "mach": mach, "alt": alt, "iota": iota}  # CA-coeff total table shape

    table_axis_labels = (Basic_labels, CA_labels, IT_labels, CA_Total_labels)
    return (data, table_axis_labels)


class AeroTable:

    """This class is for containing Aerotable data for a particular
    SimObject."""

    __ft2m = (1.0 * u.imperial.foot).to_value(u.m)  # type:ignore
    __inch2m = (1.0 * u.imperial.inch).to_value(u.m)  # type:ignore
    __deg2rad = (np.pi / 180.0)
    __lbminch2Nm = (1.0 * u.imperial.lbm * u.imperial.inch**2).to_value(u.kg * u.m**2)  # type:ignore
    __inch_sq_2_m_sq = (1.0 * u.imperial.inch**2).to_value(u.m**2)  # type:ignore

    def __init__(self, data: str|dict|MatFile, from_template: str = "", units: str = "si") -> None:
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
                data_dict, table_axis_labels = from_CMS_table(data_dict, units=units)
            case _:
                data_dict, table_axis_labels = from_default_table(data_dict, units=units)

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

        _CA_Boost_Total = data_dict.get("CA_Boost_Total", None)  # (alpha, phi, mach, alt, iota)
        _CA_Coast_Total = data_dict.get("CA_Coast_Total", None)  # (alpha, phi, mach, alt, iota)
        _CNB_Total = data_dict.get("CNB_Total", None)            # (alpha, phi, mach, iota)
        _CLMB_Total = data_dict.get("CLMB_Total", None)          # (alpha, phi, mach, iota)
        _CLNB_Total = data_dict.get("CLNB_Total", None)          # (alpha, phi, mach, iota)
        _CYB_Total = data_dict.get("CYB_Total", None)            # (alpha, phi, mach, iota)

        Sref = data_dict.get("Sref", 0)
        Lref = data_dict.get("Lref", 0)
        MRC = data_dict.get("MRC", 0)

        ############################################################
        # Initialize as DataTables
        ############################################################
        # establish default labels
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

        (Basic_labels,
         CA_labels,
         IT_labels,
         CA_Total_labels) = table_axis_labels

        # check if alpha increments are reflected across zero
        # or need to be mirrored
        flag_reflect_alpha = False
        if Basic_labels["alpha"][0] == 0:
            Basic_labels["alpha"] = self.create_mirrored_array(Basic_labels["alpha"])
            IT_labels["alpha"] = self.create_mirrored_array(IT_labels["alpha"])
            CA_Total_labels["alpha"] = self.create_mirrored_array(CA_Total_labels["alpha"])
            flag_reflect_alpha = True

        self.CA_inv = DataTable(_CA_inv, Basic_labels)
        self.CA_Basic = DataTable(_CA_Basic, Basic_labels)
        self.CA_0_Boost = DataTable(_CA_0_Boost, CA_labels)
        self.CA_0_Coast = DataTable(_CA_0_Coast, CA_labels)
        self.CA_IT = DataTable(_CA_IT, IT_labels)
        self.CYB_Basic = DataTable(_CYB_Basic, Basic_labels)
        self.CYB_IT = DataTable(_CYB_IT, IT_labels)
        self.CNB_Basic = DataTable(_CNB_Basic, Basic_labels)
        self.CNB_IT = DataTable(_CNB_IT, IT_labels)
        self.CLLB_Basic = DataTable(_CLLB_Basic, Basic_labels)
        self.CLLB_IT = DataTable(_CLLB_IT, IT_labels)
        self.CLMB_Basic = DataTable(_CLMB_Basic, Basic_labels)
        self.CLMB_IT = DataTable(_CLMB_IT, IT_labels)
        self.CLNB_Basic = DataTable(_CLNB_Basic, Basic_labels)
        self.CLNB_IT = DataTable(_CLNB_IT, IT_labels)
        self.Fin2_CN = DataTable(_Fin2_CN, IT_labels)
        self.Fin2_CBM = DataTable(_Fin2_CBM, IT_labels)
        self.Fin2_CHM = DataTable(_Fin2_CHM, IT_labels)
        self.Fin4_CN = DataTable(_Fin4_CN, IT_labels)
        self.Fin4_CBM = DataTable(_Fin4_CBM, IT_labels)
        self.Fin4_CHM = DataTable(_Fin4_CHM, IT_labels)

        self.CA_Boost_Total = DataTable(_CA_Boost_Total, CA_Total_labels)
        self.CA_Coast_Total = DataTable(_CA_Coast_Total, CA_Total_labels)
        self.CNB_Total = DataTable(_CNB_Total, IT_labels)
        self.CLMB_Total = DataTable(_CLMB_Total, IT_labels)
        self.CLNB_Total = DataTable(_CLNB_Total, IT_labels)
        self.CYB_Total = DataTable(_CYB_Total, IT_labels)

        ############################################################
        # Convert to SI units
        # TODO make input and ouput of units better...
        # NOTE: currently aero data available is in imperial units
        ############################################################
        self.Sref = Sref * self.__inch_sq_2_m_sq
        self.Lref = Lref * self.__inch2m
        self.MRC = MRC * self.__inch2m

        # MRC may be a float or array
        # TODO: maybe do this better
        if hasattr(self.MRC, "__len__"):
            self.MRC = np.asarray(self.MRC, dtype=float)
        else:
            self.MRC = float(self.MRC)

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
        Total_table_names = ["CA_Boost_Total",
                             "CA_Coast_Total",
                             "CNB_Total",
                             "CLMB_Total",
                             "CLNB_Total",
                             "CYB_Total"]
        table_names = CA_0_table_names + Basic_table_names + Iter_table_names + Total_table_names

        ############################################################
        # Table Reflections
        #   if alpha is not reflected across 0, so mirror the tables
        #   across alpha increments
        ############################################################
        if flag_reflect_alpha:
            alpha_axis = 0
            for table_name in table_names:
                _table = getattr(self, table_name)
                if not _table.isnone():
                    if "alpha" in _table.axes:
                        mirrored_table = _table.mirror_axis(alpha_axis)
                        setattr(self, table_name, mirrored_table)

        ############################################################
        # Build Total DataTables from sub-tables
        #   (Basic + Increment) tables
        ############################################################
        if self.CA_Boost_Total.isnone():
            self.CA_Boost_Total = DataTable(
                    self.CA_Basic[:, :, :, np.newaxis, np.newaxis]
                    + self.CA_0_Boost[np.newaxis, :, :, :, np.newaxis]
                    + self.CA_IT[:, :, :, np.newaxis, :],
                    axes=CA_Total_labels)
        if self.CA_Coast_Total.isnone():
            self.CA_Coast_Total = DataTable(
                    self.CA_Basic[:, :, :, np.newaxis, np.newaxis]
                    + self.CA_0_Coast[np.newaxis, :, :, :, np.newaxis]
                    + self.CA_IT[:, :, :, np.newaxis, :],
                    axes=CA_Total_labels)
        if self.CNB_Total.isnone():
            if not (self.CNB_Basic.isnone() and self.CNB_IT.isnone()):
                self.CNB_Total = DataTable(self.CNB_Basic[:, :, :, np.newaxis]
                                           + self.CNB_IT,
                                           axes=IT_labels)
        if self.CLMB_Total.isnone():
            if not (self.CLMB_Basic.isnone() and self.CLMB_IT.isnone()):
                self.CLMB_Total = DataTable(self.CLMB_Basic[:, :, :, np.newaxis]
                                            + self.CLMB_IT,
                                            axes=IT_labels)
        if self.CLNB_Total.isnone():
            if not (self.CLNB_Basic.isnone() and self.CLNB_IT.isnone()):
                self.CLNB_Total = DataTable(self.CLNB_Basic[:, :, :, np.newaxis]
                                            + self.CLNB_IT,
                                            axes=IT_labels)
        if self.CYB_Total.isnone():
            if not (self.CYB_Basic.isnone() and self.CYB_IT.isnone()):
                self.CYB_Total = DataTable(self.CYB_Basic[:, :, :, np.newaxis]
                                           + self.CYB_IT,
                                           axes=IT_labels)

        ############################################################
        # For Momementless Dynamics, the following tables are
        #   - compute CA_Boost_Total wrt. alpha
        #   - compute CA_Boost_Total wrt. alpha
        #   - compute CNB_Total wrt. alpha
        ############################################################
        if units.lower() == "si":
            delta_alpha = np.radians(0.1)
        else:
            delta_alpha = 0.1
        self.CA_Boost_Total_alpha = self.create_diff_table(table=self.CA_Boost_Total,
                                                           diff_arg="alpha",
                                                           delta_arg=delta_alpha)

        self.CA_Coast_Total_alpha = self.create_diff_table(table=self.CA_Coast_Total,
                                                           diff_arg="alpha",
                                                           delta_arg=delta_alpha)

        self.CNB_Total_alpha = self.create_diff_table(table=self.CNB_Total,
                                                      diff_arg="alpha",
                                                      delta_arg=delta_alpha)


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
        return (val_plus - val_minus) / (args_plus[diff_arg] - args_minus[diff_arg])  # type:ignore


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


    def get_CA_Boost_Total(self,
                           alpha: Optional[ArgType] = None,
                           phi: Optional[ArgType] = None,
                           mach: Optional[ArgType] = None,
                           alt: Optional[ArgType] = None,
                           iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CA_Boost_Total(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)



    def get_CA_Coast_Total(self,
                           alpha: Optional[ArgType] = None,
                           phi: Optional[ArgType] = None,
                           mach: Optional[ArgType] = None,
                           alt: Optional[ArgType] = None,
                           iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CA_Coast_Total(alpha=alpha, phi=phi, mach=mach, alt=alt, iota=iota)


    def get_CNB_Total(self,
                      alpha: Optional[ArgType] = None,
                      phi: Optional[ArgType] = None,
                      mach: Optional[ArgType] = None,
                      iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CNB_Total(alpha=alpha, phi=phi, mach=mach, iota=iota)


    def get_CLMB_Total(self,
                       alpha: Optional[ArgType] = None,
                       phi: Optional[ArgType] = None,
                       mach: Optional[ArgType] = None,
                       iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CLMB_Total(alpha=alpha, phi=phi, mach=mach, iota=iota)


    def get_CLNB_Total(self,
                       alpha: Optional[ArgType] = None,
                       phi: Optional[ArgType] = None,
                       mach: Optional[ArgType] = None,
                       iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CLNB_Total(alpha=alpha, phi=phi, mach=mach, iota=iota)


    def get_CYB_Total(self,
                      alpha: Optional[ArgType] = None,
                      phi: Optional[ArgType] = None,
                      mach: Optional[ArgType] = None,
                      iota: Optional[ArgType] = None) -> float|np.ndarray:
        return self.CYB_Total(alpha=alpha, phi=phi, mach=mach, iota=iota)


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
