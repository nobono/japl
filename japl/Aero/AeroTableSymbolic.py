from sympy import Function
from japl import AeroTable
from japl.Util.Matlab import MatFile



class AeroTableSymbolic:

    """This is the Symbolic mirror of the AeroTable module
    which can be used for creating models from symblic expressions."""


    def __init__(self, data: str|dict|MatFile) -> None:
        self.aerotable = AeroTable(data)
        self.modules = {
                "aerotable_get_CA_Boost_Total": self.aerotable.get_CA_Boost_Total,
                "aerotable_get_CA_Coast_Total": self.aerotable.get_CA_Coast_Total,
                "aerotable_get_CNB_Total": self.aerotable.get_CNB_Total,
                "aerotable_get_CLMB_Total": self.aerotable.get_CLMB_Total,
                "aerotable_get_CLNB_Total": self.aerotable.get_CLNB_Total,
                "aerotable_get_CYB_Total": self.aerotable.get_CYB_Total,
                "aerotable_get_MRC": self.aerotable.get_MRC,
                "aerotable_get_Sref": self.aerotable.get_Sref,
                "aerotable_get_Lref": self.aerotable.get_Lref,
                }
        self.get_CA_Boost_Total = Function("aerotable_get_CA_Boost_Total")
        self.get_CA_Coast_Total = Function("aerotable_get_CA_Coast_Total")
        self.get_CNB_Total = Function("aerotable_get_CNB_Total")
        self.get_CLMB_Total = Function("aerotable_get_CLMB_Total")
        self.get_CLNB_Total = Function("aerotable_get_CLNB_Total")
        self.get_CYB_Total = Function("aerotable_get_CYB_Total")
        self.get_MRC = Function("aerotable_get_MRC")
        self.get_Sref = Function("aerotable_get_Sref")
        self.get_Lref = Function("aerotable_get_Lref")
