from typing import Optional
from pathlib import Path
import numpy as np
from astropy.units import Unit
from astropy import units as u
from scipy.optimize import minimize_scalar
from japl.Util.Matlab import MatFile
from japl.DataTable.DataTable import DataTable



class Increments:
    alpha = np.empty([])
    beta = np.empty([])
    phi = np.empty([])
    mach = np.empty([])
    alt = np.empty([])
    iota = np.empty([])
    iota_prime = np.empty([])

    nalpha = 0
    nbeta = 0
    nphi = 0
    nmach = 0
    nalt = 0
    niota = 0
    niota_prime = 0


class AeroTable:
    increments: Increments
    stages: list["AeroTable"]
    stage_id: int
    is_stage: bool
    Sref: float
    Lref: float
    MRC: float
    CA: DataTable
    CA_Boost: DataTable
    CA_Coast: DataTable
    CNB: DataTable
    CYB: DataTable
    CA_Boost_alpha: DataTable
    CA_Coast_alpha: DataTable
    CNB_alpha: DataTable

    def __new__(cls, path: str = "",
                ignore_units: bool = False,
                angle_units: Unit = u.rad,  # type:ignore
                length_units: Unit = u.m,  # type:ignore
                sref_units: Unit = u.m**2,  # type:ignore
                lref_units: Unit = u.m,  # type:ignore
                mrc_units: Unit = u.m,  # type:ignore
                ):

        obj = super().__new__(cls)
        obj.increments = Increments()
        obj.stages = []
        obj.stage_id = 0
        obj.is_stage = True
        if path:
            matfile = MatFile(path)
            obj._build_from_matfile(matfile,
                                    ignore_units=ignore_units,
                                    angle_units=angle_units,
                                    length_units=length_units,
                                    sref_units=sref_units,
                                    lref_units=lref_units,
                                    mrc_units=mrc_units)
        return obj


    # def __init__(self, path: str = "") -> None:
    #     self.stages: list[AeroTable] = []
    #     self.stage_id: int = 0
    #     self.is_stage: bool = True


    def _build_from_matfile(self, file: MatFile,
                            ignore_units: bool = False,
                            angle_units: Unit = u.rad,  # type:ignore
                            length_units: Unit = u.m,  # type:ignore
                            sref_units: Optional[Unit] = None,  # type:ignore
                            lref_units: Optional[Unit] = None,  # type:ignore
                            mrc_units: Optional[Unit] = None,  # type:ignore
                            ):
        """Attempts to auto-build AeroTable from provided MatFile.
        makes assumptions about each DataTables' axes / axes order.
        Unless specific unit arguments are provided, SI units will
        be assumed.

        -------------------------------------------------------------------

        Parameters:
            file: a MatFile object

            angle_units: astropy unit for all angles

            length_units: astropy unit for all lengths

        Returns:
            an AeroTable

        -------------------------------------------------------------------
        """
        # NOTE: this will search the MatFile for any attributes unit related
        unit_info = file.findall("unit", case_sensitive=False)

        if ignore_units:
            angle_conv_const = 1.0
            length_conv_const = 1.0
            area_conv_const = 1.0
            sref_conv_const = 1.0
            lref_conv_const = 1.0
            mrc_conv_const = 1.0
        else:
            angle_conv_const = float((1.0 * angle_units).si.to_value())  # type:ignore
            length_conv_const = float((1.0 * length_units).si.to_value())  # type:ignore
            area_conv_const = float((1.0 * length_units**2).si.to_value())  # type:ignore
            if sref_units:
                sref_conv_const = float((1.0 * sref_units).si.to_value())  # type:ignore
            else:
                area_conv_const = float((1.0 * length_units**2).si.to_value())  # type:ignore
                sref_conv_const = area_conv_const
            if lref_units:
                lref_conv_const = float((1.0 * lref_units).si.to_value())  # type:ignore
            else:
                lref_conv_const = length_conv_const
            if mrc_units:
                mrc_conv_const = float((1.0 * mrc_units).si.to_value())  # type:ignore
            else:
                mrc_conv_const = length_conv_const

        # -----------------------------------------------------------------
        # NOTE: order of increments.__dict__ is preserved
        # from the order of __setattr__ below:
        # -----------------------------------------------------------------
        default = np.array([])
        self.increments.alpha = file.find(["alpha"], default=default)
        self.increments.phi = file.find(["phi"], default=default)
        self.increments.mach = file.find(["mach"], default=default)
        self.increments.alt = file.find(["alt", "altitude"], default=default)
        self.increments.iota = file.find(["iota"], default=default)
        self.increments.alpha = self.increments.alpha * angle_conv_const
        self.increments.phi = self.increments.phi * angle_conv_const
        self.increments.alt = self.increments.alt * length_conv_const
        self.increments.iota = self.increments.iota * angle_conv_const

        possible_axes = self.increments.__dict__

        CA_Boost = file.find(["CA_Boost", "CA_Powered"])
        CA_Coast = file.find(["CA_Coast", "CA_Unpowered"])
        CNB = file.find(["CN", "CNB"])
        CLMB = file.find(["CLM", "CLMB"])
        CLNB = file.find(["CLN", "CLNB"])
        CYB = file.find(["CY", "CYB"])

        CA_Basic = file.find(["CA_Basic"], default=default)
        CA_0_Boost = file.find(["CA_0_Boost", "CA0_Boost"])
        CA_0_Coast = file.find(["CA_0_Coast", "CA0_Coast"])
        CA_IT = file.find(["CA_IT"])

        CNB_Basic = file.find(["CNB_Basic", "CN_Basic"])
        CNB_IT = file.find(["CNB_IT", "CN_IT"])

        CLMB_Basic = file.find(["CLMB_Basic", "CLM_Basie"])
        CLMB_IT = file.find(["CLMB_IT", "CLM_IT"])
        CLNB_Basic = file.find(["CLNB_Basic", "CLN_Basic"])
        CLNB_IT = file.find(["CLNB_IT", "CLN_IT"])

        CYB_Basic = file.find(["CYB_Basic", "CY_Basic"])
        CYB_IT = file.find(["CYB_IT", "CY_IT"])

        tables = {
                  "CA_Boost": CA_Boost,
                  "CA_Coast": CA_Coast,
                  "CNB": CNB,
                  "CLMB": CLMB,
                  "CLNB": CLNB,
                  "CYB": CYB,
                  "CA_Basic": CA_Basic,
                  "CA_0_Boost": CA_0_Boost,
                  "CA_0_Coast": CA_0_Coast,
                  "CA_IT": CA_IT,
                  "CNB_Basic": CNB_Basic,
                  "CNB_IT": CNB_IT,
                  "CLMB_Basic": CLMB_Basic,
                  "CLMB_IT": CLMB_IT,
                  "CLNB_Basic": CLNB_Basic,
                  "CLNB_IT": CLNB_IT,
                  "CYB_Basic": CYB_Basic,
                  "CYB_IT": CYB_IT,
                  }

        for key, val in tables.items():
            if hasattr(val, "table"):
                tables[key] = val.table

        # axes_dims = [len(i) for i in unordered_axes.values()]  # ensure possible table axes are unique
        # if len(axes_dims) == len(set(axes_dims)):

        # -----------------------------------------------------------------
        # create basic tables
        # will return None if table passed is None
        # -----------------------------------------------------------------
        CA_Boost = self._deduce_datatable(tables["CA_Boost"], possible_axes=possible_axes)
        CA_Coast = self._deduce_datatable(tables["CA_Coast"], possible_axes=possible_axes)
        CNB = self._deduce_datatable(tables["CNB"], possible_axes=possible_axes)
        CLMB = self._deduce_datatable(tables["CLMB"], possible_axes=possible_axes)
        CLNB = self._deduce_datatable(tables["CLNB"], possible_axes=possible_axes)
        CYB = self._deduce_datatable(tables["CYB"], possible_axes=possible_axes)

        CA_Basic = self._deduce_datatable(tables["CA_Basic"], possible_axes=possible_axes)
        CA_0_Boost = self._deduce_datatable(tables["CA_0_Boost"], possible_axes=possible_axes)
        CA_0_Coast = self._deduce_datatable(tables["CA_0_Coast"], possible_axes=possible_axes)
        CA_IT = self._deduce_datatable(tables["CA_IT"], possible_axes=possible_axes)

        CNB_Basic = self._deduce_datatable(tables["CNB_Basic"], possible_axes=possible_axes)
        CNB_IT = self._deduce_datatable(tables["CNB_IT"], possible_axes=possible_axes)

        CLMB_Basic = self._deduce_datatable(tables["CLMB_Basic"], possible_axes=possible_axes)
        CLMB_IT = self._deduce_datatable(tables["CLMB_IT"], possible_axes=possible_axes)
        CLNB_Basic = self._deduce_datatable(tables["CLNB_Basic"], possible_axes=possible_axes)
        CLNB_IT = self._deduce_datatable(tables["CLNB_IT"], possible_axes=possible_axes)

        CYB_Basic = self._deduce_datatable(tables["CYB_Basic"], possible_axes=possible_axes)
        CYB_IT = self._deduce_datatable(tables["CYB_IT"], possible_axes=possible_axes)

        if not (CA_Boost.isnone() and CA_Coast.isnone()):
            self.CA_Boost = CA_Boost
            self.CA_Coast = CA_Coast
        elif not (CA_Basic.isnone() and CA_0_Boost.isnone()
                  and CA_0_Coast.isnone() and CA_IT.isnone()):
            self.CA_Boost = CA_Basic + CA_0_Boost + CA_IT
            self.CA_Coast = CA_Basic + CA_0_Coast + CA_IT
        else:
            self.CA_Boost = DataTable(None, {})
            self.CA_Coast = DataTable(None, {})

        if not CNB.isnone():
            self.CNB = CNB
        elif not (CNB_Basic.isnone() and CNB_IT.isnone()):
            self.CNB = CNB_Basic + CNB_IT
        else:
            self.CNB = DataTable(None, {})

        if not CLMB.isnone():
            self.CLMB = CLMB
        elif not (CLMB_Basic.isnone() and CLMB_IT.isnone()):
            self.CLMB = CLMB_Basic + CLMB_IT
        else:
            self.CLMB = DataTable(None, {})

        if not CLNB.isnone():
            self.CLNB = CLNB
        elif not (CLNB_Basic.isnone() and CLNB_IT.isnone()):
            self.CLNB = CLNB_Basic + CLNB_IT
        else:
            self.CLNB = DataTable(None, {})

        if not CYB.isnone():
            self.CYB = CYB
        elif not (CYB_Basic.isnone() and CYB_IT.isnone()):
            self.CYB = CYB_Basic + CYB_IT
        else:
            self.CYB = DataTable(None, {})

        # -----------------------------------------------------------------
        # create diff tables
        # -----------------------------------------------------------------
        units = "si"
        if units.lower() == "si":
            delta_alpha = np.radians(0.1)
        else:
            delta_alpha = 0.1
        self.CA_Boost_alpha = self.CA_Boost.create_diff_table(diff_arg="alpha", delta_arg=delta_alpha)
        self.CA_Coast_alpha = self.CA_Coast.create_diff_table(diff_arg="alpha", delta_arg=delta_alpha)
        self.CNB_alpha = self.CNB.create_diff_table(diff_arg="alpha", delta_arg=delta_alpha)

        # -----------------------------------------------------------------
        # mirror axes
        # -----------------------------------------------------------------
        if not self.CA_Boost.isnone():
            self.CA_Boost = self.CA_Boost.mirror_axis("alpha")
        if not self.CA_Coast.isnone():
            self.CA_Coast = self.CA_Coast.mirror_axis("alpha")
        if not self.CNB.isnone():
            self.CNB = self.CNB.mirror_axis("alpha")
        if not self.CYB.isnone():
            self.CYB = self.CYB.mirror_axis("alpha")

        self.CA_Boost_alpha = self.CA_Boost_alpha.mirror_axis("alpha")
        self.CA_Coast_alpha = self.CA_Coast_alpha.mirror_axis("alpha")
        self.CNB_alpha = self.CNB_alpha.mirror_axis("alpha")

        # -----------------------------------------------------------------
        # Ensure axes order
        # -----------------------------------------------------------------
        axes_order = ["alpha", "beta", "phi", "mach", "alt", "iota"]
        if not CA_Boost.isnone():
            self.CA_Boost = self.CA_Boost.swap_to_label_order(axes_order)
        if not CA_Coast.isnone():
            self.CA_Coast = self.CA_Coast.swap_to_label_order(axes_order)
        if not CNB.isnone():
            self.CNB = self.CNB.swap_to_label_order(axes_order)
        if not CYB.isnone():
            self.CYB = self.CYB.swap_to_label_order(axes_order)
        if not self.CA_Boost_alpha.isnone():
            self.CA_Boost_alpha = self.CA_Boost_alpha.swap_to_label_order(axes_order)
        if not self.CA_Coast_alpha.isnone():
            self.CA_Coast_alpha = self.CA_Coast_alpha.swap_to_label_order(axes_order)
        if not self.CNB_alpha.isnone():
            self.CNB_alpha = self.CNB_alpha.swap_to_label_order(axes_order)

        # -----------------------------------------------------------------
        # other values
        # -----------------------------------------------------------------
        self.Sref = file.find(["sref"], case_sensitive=False, default=0)
        self.Lref = file.find(["lref"], case_sensitive=False, default=0)
        self.MRC = file.find(["MRC", "MRP"], case_sensitive=False, default=0)

        self.Sref *= sref_conv_const
        self.Lref *= lref_conv_const
        self.MRC *= mrc_conv_const


    @staticmethod
    def _deduce_datatable(table: np.ndarray, possible_axes: dict) -> DataTable:
        """Attempts to auto-detect the order of DataTable axes from a list of
        unordered possible axes.

        Limitations:
            - axis order cannot be deduced if table dimensions are not unique.
              i.e. table.shape = (1, 2, 3, 3).
            - in this case, suggested_order is used to assume what axes have
              priority over others.

        -------------------------------------------------------------------

        Parameters:
            table: numpy array

            unordered_axes: dict of unordered axes. the order of axis keys
                            aid deduction when dealing with table axes
                            that are not unique.

        Returns:
            a DataTable or None

        -------------------------------------------------------------------
        """
        if table is None:
            return DataTable(None, {})

        ordered_axes = []
        used_axes = []
        for ndim in table.shape:
            for key, val in possible_axes.items():
                if len(val) == ndim and key not in used_axes:
                    ordered_axes += [(key, val.astype(float))]
                    used_axes += [key]
                    break
        axes = dict(ordered_axes)
        return DataTable(table, axes)


    def add_stage(self, aerotable: "AeroTable") -> None:
        """Adds a child AeroTable object as an ordered child
        of this object."""
        self.is_stage = False
        aerotable.is_stage = True
        self.stages += [aerotable]

    def set_stage(self, stage: int) -> None:
        """Set the current stage index for the aerotable. This is
        so that \"get_stage()\" will return the corresponding aerotable."""
        if int(stage) >= len(self.stages):
            raise Exception(f"cannot access stage {int(stage)} "
                            f"for container of size {len(self.stages)}")
        self.stage_id = int(stage)

    def get_stage(self) -> "AeroTable":
        """Returns the current aerotable corresponding to the stage_id."""
        if self.is_stage:
            return self
        else:
            return self.stages[self.stage_id]

    def get_Sref(self):
        return self.get_stage().Sref

    def get_Lref(self):
        return self.get_stage().Lref

    def get_MRC(self):
        return self.get_stage().MRC

    def get_CA(self, *args, **kwargs):
        return self.get_stage().CA(*args, **kwargs)

    def get_CA_Boost(self, *args, **kwargs):
        return self.get_stage().CA_Boost(*args, **kwargs)

    def get_CA_Coast(self, *args, **kwargs):
        return self.get_stage().CA_Coast(*args, **kwargs)

    def get_CNB(self, *args, **kwargs):
        return self.get_stage().CNB(*args, **kwargs)

    def get_CYB(self, *args, **kwargs):
        return self.get_stage().CYB(*args, **kwargs)

    def get_CA_Boost_alpha(self, *args, **kwargs):
        return self.get_stage().CA_Boost_alpha(*args, **kwargs)

    def get_CA_Coast_alpha(self, *args, **kwargs):
        return self.get_stage().CA_Coast_alpha(*args, **kwargs)

    def get_CNB_alpha(self, *args, **kwargs):
        return self.get_stage().CNB_alpha(*args, **kwargs)

    def inv_aerodynamics(self,
                         thrust: float,
                         acc_cmd: float,
                         beta: float,
                         dynamic_pressure: float,
                         mass: float,
                         alpha: float,
                         phi: Optional[float] = None,
                         mach: Optional[float] = None,
                         alt: Optional[float] = None,
                         iota: Optional[float] = None) -> float:
        stage = self.get_stage()
        alpha_tol = np.radians(0.01)
        alpha_max = max(stage.increments.alpha)
        Sref = stage.get_Sref()

        alpha_last = -1000
        count = 0

        boosting = (thrust > 0.0)

        # gradient search
        while ((abs(alpha - alpha_last) > alpha_tol) and (count < 10)):  # type:ignore
            count += 1
            alpha_last = alpha
            if boosting:
                CA = stage.get_CA_Boost(alpha=alpha, beta=beta, phi=phi,
                                        mach=mach, alt=alt, iota=iota)
            else:
                CA = stage.get_CA_Coast(alpha=alpha, beta=beta, phi=phi,
                                        mach=mach, alt=alt, iota=iota)
            CN = stage.get_CNB(alpha=alpha, beta=beta, phi=phi,
                               mach=mach, alt=alt, iota=iota)
            if boosting:
                CA_alpha = stage.get_CA_Boost_alpha(alpha=alpha, beta=beta, phi=phi,
                                                    mach=mach, alt=alt, iota=iota)
            else:
                CA_alpha = stage.get_CA_Coast_alpha(alpha=alpha, beta=beta, phi=phi,
                                                    mach=mach, alt=alt, iota=iota)
            CN_alpha = stage.get_CNB_alpha(alpha=alpha, beta=beta, phi=phi,
                                           mach=mach, alt=alt, iota=iota)

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

    def ld_guidance(self,
                    alpha: float,
                    beta: Optional[float] = None,
                    phi: Optional[float] = None,
                    mach: Optional[float] = None,
                    alt: Optional[float] = None,
                    iota: Optional[float] = None,
                    thrust: float = 0):
        stage = self.get_stage()
        alpha_max = stage.CNB.axes["alpha"].max()
        alpha_min = stage.CNB.axes["alpha"].min()
        # if alpha > alpha_max or alpha < alpha_min:
        #     print("alpha clipping....")

        def ld_ratio(_alpha):
            stage = self.get_stage()
            boosting = (thrust > 0.0)
            if boosting:
                CA = stage.get_CA_Boost(alpha=abs(alpha), beta=beta, phi=phi,
                                        mach=mach, alt=alt, iota=iota)
            else:
                CA = stage.get_CA_Coast(alpha=abs(alpha), beta=beta, phi=phi,
                                        mach=mach, alt=alt, iota=iota)
            CN = stage.get_CNB(alpha=_alpha, beta=beta, phi=phi,
                               mach=mach, alt=alt, iota=iota)
            cosa = np.cos(_alpha)
            sina = np.sin(_alpha)
            CL = (CN * cosa) - (CA * sina)
            CD = (CN * sina) + (CA * cosa)
            ratio = -CL / CD
            return ratio
        result = minimize_scalar(ld_ratio, bounds=(alpha_min, alpha_max), method='bounded')
        optimal_alpha = result.x  # type:ignore
        # ld_max = -result.fun  # type:ignore
        # optimal CL, CD
        boosting = (thrust > 0.0)
        if boosting:
            CA = stage.get_CA_Boost(alpha=abs(alpha), beta=beta, phi=phi, mach=mach, alt=alt, iota=iota)  # type:ignore
        else:
            CA = stage.get_CA_Coast(alpha=abs(alpha), beta=beta, phi=phi, mach=mach, alt=alt, iota=iota)  # type:ignore
        CN = stage.get_CNB(alpha=optimal_alpha, beta=beta, phi=phi, mach=mach, alt=alt, iota=iota)
        cosa = np.cos(optimal_alpha)
        sina = np.sin(optimal_alpha)
        opt_CL = (CN * cosa) - (CA * sina)
        opt_CD = (CN * sina) + (CA * cosa)

        optimal_alpha = np.clip(optimal_alpha, alpha_min, alpha_max)
        return opt_CL, opt_CD, optimal_alpha

    def _set_table_check(self):
        """handle errors when setting tables."""
        if not self.is_stage:
            raise Exception("cannot set tables of AeroTableable container.",
                            "This object is just a container for multiple"
                            "AeroTableable's within the \"stages\" attribute.")


# print(Path("./aerodata/aeromodel_psb.mat").absolute().__str__())
# from japl.global_opts import get_root_dir
# aero = AeroTable(Path(f"{get_root_dir()}/aerodata/aeromodel_psb.mat").absolute().__str__())
