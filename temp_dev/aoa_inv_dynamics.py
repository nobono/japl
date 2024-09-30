from sympy import symbols, Symbol, Matrix
from sympy import Function
from sympy import cos, sin, tan
from sympy import atan2
from sympy import sqrt
from sympy import diff
import sympy as sp



t = Symbol("t")
mass = Symbol("mass")
thrust = Symbol("thrust")
Sref = Symbol("Sref")
q = Symbol("q")
alpha = Symbol("alpha")

CN = Function("CN")(alpha)  # type:ignore
CA = Function("CA")(alpha)  # type:ignore
CL = Function("CL")(alpha)  # type:ignore
CD = Function("CD")(alpha)  # type:ignore
CN_alpha = Symbol("CN_alpha")
CA_alpha = Symbol("CA_alpha")

##################################################
# getAeroCoeffs
##################################################
CL = (CN * cos(alpha)) - (CA * sin(alpha))  # type:ignore
CD = (CN * sin(alpha)) + (CA * cos(alpha))  # type:ignore

CL_alpha = CL.diff(alpha)  # .subs(defs)
CD_alpha = CD.diff(alpha)  # .subs(defs)

defs = [
        (diff(CN, alpha), CN_alpha),
        (diff(CA, alpha), CA_alpha),
        ]

##################################################
# ComputeAoACommand
##################################################
# acceleration normal to flight path
aN = (CL * q * Sref / mass) + (thrust / mass) * sin(alpha)  # type:ignore
aN_alpha = aN.diff(alpha)


# subs = [
#         (CL.diff(alpha), "CL_alpha")
#         ]
