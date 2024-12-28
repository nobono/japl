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

    def __new__(cls, path: str = "") -> None:
        obj = super().__new__(cls)
        obj.increments = Increments()
        obj.stages = []
        obj.stage_id = 0
        obj.is_stage = True
        if path:
            matfile = MatFile(path)
            obj._build_from_matfile(matfile)


    # def __init__(self, path: str = "") -> None:
    #     self.stages: list[AeroTable] = []
    #     self.stage_id: int = 0
    #     self.is_stage: bool = True


    def _build_from_matfile(self, file: MatFile,
                            angle_units: Unit = u.rad,  # type:ignore
                            length_units: Unit = u.m,  # type:ignore
                            ) -> "AeroTable":
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
        unit_info = file.findall("unit", case_sensitive=False)

        angle_conv_const = float((1.0 * angle_units).si.to_value())  # type:ignore
        length_conv_const = float((1.0 * length_units).si.to_value())  # type:ignore
        area_conv_const = float((1.0 * length_units**2).si.to_value())  # type:ignore

        # -----------------------------------------------------------------
        # NOTE: order of increments.__dict__ is preserved
        # from the order of __setattr__ below:
        # -----------------------------------------------------------------
        self.increments.alpha = file.Alpha * angle_conv_const  # type:ignore
        self.increments.phi = file.Phi * angle_conv_const  # type:ignore
        self.increments.mach = file.Mach  # type:ignore
        self.increments.alt = file.Alt * length_conv_const  # type:ignore
        self.increments.iota = file.Iota * angle_conv_const  # type:ignore

        possible_axes = self.increments.__dict__

        _CA_Basic = file.find(["CA_Basic"]).table
        _CA_0_Boost = file.find(["CA_0_Boost", "CA0_Boost"]).table
        _CA_0_Coast = file.find(["CA_0_Coast", "CA0_Coast"]).table
        _CA_IT = file.find(["CA_IT"]).table

        _CNB_Basic = file.find(["CNB_Basic", "CN_Basic"]).table
        _CNB_IT = file.find(["CNB_IT", "CN_IT"]).table

        _CLMB_Basic = file.find(["CLMB_Basic", "CLM_Basic"]).table
        _CLMB_IT = file.find(["CLMB_IT", "CLM_IT"]).table
        _CLNB_Basic = file.find(["CLNB_Basic", "CLN_Basic"]).table
        _CLNB_IT = file.find(["CLNB_IT", "CLN_IT"]).table

        _CYB_Basic = file.find(["CYB_Basic", "CY_Basic"]).table
        _CYB_IT = file.find(["CYB_IT", "CY_IT"]).table

        # axes_dims = [len(i) for i in unordered_axes.values()]  # ensure possible table axes are unique
        # if len(axes_dims) == len(set(axes_dims)):

        # -----------------------------------------------------------------
        # create basic tables
        # -----------------------------------------------------------------
        CA_Basic = self._deduce_datatable(_CA_Basic, possible_axes=possible_axes)
        CA_0_Boost = self._deduce_datatable(_CA_0_Boost, possible_axes=possible_axes)
        CA_0_Coast = self._deduce_datatable(_CA_0_Coast, possible_axes=possible_axes)
        CA_IT = self._deduce_datatable(_CA_IT, possible_axes=possible_axes)

        CNB_Basic = self._deduce_datatable(_CNB_Basic, possible_axes=possible_axes)
        CNB_IT = self._deduce_datatable(_CNB_IT, possible_axes=possible_axes)

        CLMB_Basic = self._deduce_datatable(_CLMB_Basic, possible_axes=possible_axes)
        CLMB_IT = self._deduce_datatable(_CLMB_IT, possible_axes=possible_axes)
        CLNB_Basic = self._deduce_datatable(_CLNB_Basic, possible_axes=possible_axes)
        CLNB_IT = self._deduce_datatable(_CLNB_IT, possible_axes=possible_axes)

        CYB_Basic = self._deduce_datatable(_CYB_Basic, possible_axes=possible_axes)
        CYB_IT = self._deduce_datatable(_CYB_IT, possible_axes=possible_axes)

        self.CA_Boost = CA_Basic + CA_0_Boost + CA_IT
        self.CA_Coast = CA_Basic + CA_0_Coast + CA_IT
        self.CNB = CNB_Basic + CNB_IT
        self.CLMB = CLMB_Basic + CLMB_IT
        self.CLNB = CLNB_Basic + CLNB_IT
        self.CYB = CYB_Basic + CYB_IT

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
        # other values
        # -----------------------------------------------------------------
        self.Sref = file.find(["sref"], case_sensitive=False)
        self.Lref = file.find(["lref"], case_sensitive=False)
        self.MRC = file.find(["MRC", "MRP"], case_sensitive=False)

        self.Sref *= area_conv_const
        self.Lref *= length_conv_const


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
        ordered_axes = []
        used_axes = []
        for ndim in table.shape:
            for key, val in possible_axes.items():
                if len(val) == ndim and key not in used_axes:
                    ordered_axes += [(key, val)]
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
from japl.global_opts import get_root_dir
aero = AeroTable(Path(f"{get_root_dir()}/aerodata/aeromodel_psb.mat").absolute().__str__())

# AEROMODEL_DIR = "aerodata/"
# stage_1_aero = AeroTableable(AEROMODEL_DIR + "stage_1_aero.mat", from_template="orion")
# stage_2_aero = AeroTableable(AEROMODEL_DIR + "stage_2_aero.mat", from_template="orion")
# aerotable = AeroTable()
# stage_1_aero = AeroTable()
# stage_2_aero = AeroTable()

# stage_1_aero.CA_Boost = aero1.CA_Boost
# stage_1_aero.CA_Coast = aero1.CA_Coast
# stage_1_aero.CNB = aero1.CNB
# stage_1_aero.CYB = aero1.CYB
# stage_1_aero.CA_Boost_alpha = aero1.CA_Boost_alpha
# stage_1_aero.CA_Coast_alpha = aero1.CA_Coast_alpha
# stage_1_aero.CNB_alpha = aero1.CNB_alpha
# stage_1_aero.increments.alpha = aero1.alpha
# stage_1_aero.increments.beta = aero1.beta
# stage_1_aero.increments.mach = aero1.mach
# stage_1_aero.increments.alt = aero1.alt
# stage_1_aero.Sref = aero1.Sref
# stage_1_aero.Lref = aero1.Lref
# stage_1_aero.MRC = aero1.MRP

# stage_2_aero.CA_Boost = aero2.CA_Boost
# stage_2_aero.CA_Coast = aero2.CA_Coast
# stage_2_aero.CNB = aero2.CNB
# stage_2_aero.CYB = aero2.CYB
# stage_2_aero.CA_Boost_alpha = aero2.CA_Boost_alpha
# stage_2_aero.CA_Coast_alpha = aero2.CA_Coast_alpha
# stage_2_aero.CNB_alpha = aero2.CNB_alpha
# stage_2_aero.increments.alpha = aero2.alpha
# stage_2_aero.increments.beta = aero2.beta
# stage_2_aero.increments.mach = aero2.mach
# stage_2_aero.increments.alt = aero2.alt
# stage_2_aero.Sref = aero2.Sref
# stage_2_aero.Lref = aero2.Lref
# stage_2_aero.MRC = aero2.MRP

# aerotable.add_stage(stage_1_aero)
# aerotable.add_stage(stage_2_aero)
