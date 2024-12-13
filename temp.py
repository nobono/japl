from sympy import symbols
from sympy import Matrix
from sympy import MatrixSymbol
from sympy import Symbol
from japl.CodeGen import pycode
from japl.CodeGen import ccode
from japl.CodeGen import Builder
from japl.CodeGen import FileBuilder
from japl.CodeGen import CFileBuilder
from japl.CodeGen import ModuleBuilder
from japl.CodeGen import CodeGenerator
from japl.CodeGen import JaplFunction
from japl.CodeGen.Ast import Kwargs
from japl.CodeGen.Ast import Dict

from japl.CodeGen.JaplFunction import numbered_symbols, get_lang_types, Variable



class func(JaplFunction):
    pass


a, b = symbols("a, b")
c, d = symbols("c, d")
A = Matrix([c, d])
B = Matrix([c, d])
# A.name = "A"
# B.name = "B"
# A = MatrixSymbol("A", 3, 1)
# B = MatrixSymbol("B", 3, 1)
# f = func(A, B=B, C=A)

# k = Kwargs(a=a)
# print(k.type)

f = func(a, b=b)
f.set_body(a + b)
f._build_function('c')

print(ccode(f.codegen_function_proto))
print(ccode(f.codegen_function_def))

# TODO pycode func def needs work
print(pycode(f.codegen_function_def))

# print(pycode(f))
# print(ccode(f))
