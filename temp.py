# from sympy import symbols
# from sympy import Matrix
# from sympy import true, false
# from sympy import cse
# from sympy import MatrixSymbol
# from sympy import Symbol
# from sympy import Function
# from sympy.codegen.ast import CodeBlock, Element
# from sympy.codegen.ast import Type
# from sympy.codegen.ast import String
# from sympy.matrices.expressions.matexpr import MatrixElement
# from sympy.printing.c import Assignment, Declaration
# from japl.CodeGen import pycode
# from japl.CodeGen import ccode
# from japl.CodeGen import Builder
# from japl.CodeGen import FileBuilder
# from japl.CodeGen import CFileBuilder
# from japl.CodeGen import ModuleBuilder
# from japl.CodeGen import CodeGenerator
# from japl.CodeGen import JaplFunction
# from japl.CodeGen.Ast import CTypes, JaplType, JaplTypes, Kwargs, CType, PyType
# from japl.CodeGen.Ast import convert_symbols_to_variables
# from japl.CodeGen.Ast import Dict
# from japl.CodeGen.JaplFunction import numbered_symbols, get_lang_types, Variable
# from japl.CodeGen.Util import optimize_expression
# from japl.Library.Earth.Earth import Earth
# from sympy import sin, cos
from japl import Model



# t = symbols("t")
# dt = symbols("dt")

# gacc = Symbol("gacc", real=True)
# omega_e = Earth.omega
# x = Function("x", real=True)(t)
# y = Function("y", real=True)(t)
# z = Function("z", real=True)(t)
# vx = Function("vx", real=True)(t)
# vy = Function("vy", real=True)(t)
# vz = Function("vz", real=True)(t)
# pos = Matrix([x, y, z])  # eci position
# vel = Matrix([vx, vy, vz])

# q0 = Function("q0", real=True)(t)
# q1 = Function("q1", real=True)(t)
# q2 = Function("q2", real=True)(t)
# q3 = Function("q3", real=True)(t)

from mtest import model as mod
# from japl import Model
