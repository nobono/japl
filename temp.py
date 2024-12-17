from sympy import symbols
from sympy import Matrix
from sympy.codegen.ast import CodeBlock, Element
from sympy.codegen.ast import Type
from sympy.codegen.ast import String
from sympy import true, false
from sympy import cse
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy import MatrixSymbol
from sympy import Symbol
from sympy.printing.c import Assignment, Declaration
from japl.CodeGen import pycode
from japl.CodeGen import ccode
from japl.CodeGen import Builder
from japl.CodeGen import FileBuilder
from japl.CodeGen import CFileBuilder
from japl.CodeGen import ModuleBuilder
from japl.CodeGen import CodeGenerator
from japl.CodeGen import JaplFunction
from japl.CodeGen.Ast import CTypes, JaplType, JaplTypes, Kwargs, CType, PyType
from japl.CodeGen.Ast import convert_symbols_to_variables
from japl.CodeGen.Ast import Dict


from japl.CodeGen.JaplFunction import numbered_symbols, get_lang_types, Variable
from japl.CodeGen.Util import optimize_expression



# class JaplAssignment(Assignment):

#     """This class is an augmentation of sympy Assignment where
#     the only differnce is JaplAssignment allows for Variable Declarations
#     as the lhs of the assignment."""

#     @classmethod
#     def _check_args(cls, lhs, rhs):
#         """ Check arguments to __new__ and raise exception if any problems found.

#         Derived classes may wish to override this.
#         """
#         from sympy.matrices.expressions.matexpr import (
#             MatrixElement, MatrixSymbol)
#         from sympy.tensor.indexed import Indexed
#         from sympy.tensor.array.expressions import ArrayElement

#         # Tuple of things that can be on the lhs of an assignment
#         assignable = (Symbol, MatrixSymbol, MatrixElement, Indexed, Element, Variable,
#                       ArrayElement, Declaration)
#         if not isinstance(lhs, assignable):
#             raise TypeError("Cannot assign to lhs of type %s." % type(lhs))

#         # Indexed types implement shape, but don't define it until later. This
#         # causes issues in assignment validation. For now, matrices are defined
#         # as anything with a shape that is not an Indexed
#         lhs_is_mat = hasattr(lhs, 'shape') and not isinstance(lhs, Indexed)
#         rhs_is_mat = hasattr(rhs, 'shape') and not isinstance(rhs, Indexed)

#         # If lhs and rhs have same structure, then this assignment is ok
#         if lhs_is_mat:
#             if not rhs_is_mat:
#                 raise ValueError("Cannot assign a scalar to a matrix.")
#             elif lhs.shape != rhs.shape:
#                 raise ValueError("Dimensions of lhs and rhs do not align.")
#         elif rhs_is_mat and not lhs_is_mat:
#             raise ValueError("Cannot assign a matrix to a scalar.")


t, dt = symbols("t, dt")
a, b = symbols("a, b")
c, d = symbols("c, d")
A = Matrix([a + 1, b + 1])
B = Matrix([c + 2, d + 2])


class func(JaplFunction):
    # expr = A + B / a
    expr = b * A * (a + 2)


code_type = 'py'
autocode = ccode if code_type == 'c' else pycode
# f = func(a, b=b, c=c)
# f = func(a, A)
# f._build_function(code_type, use_parallel=False, use_std_args=True)
# print(autocode(f.get_def()))


X = Matrix([a, b])
U = Matrix([c])
S = Matrix([])


class dynamics(JaplFunction):
    expr = Matrix([1, a + 2])


f = func(t, X, U, S, dt)
f._build_function(code_type, use_parallel=False, use_std_args=True)
print(autocode(f.function_def))


# f.set_body(expr, code_type=code_type)
# f._build_function(code_type)

# print(ccode(f.function_proto))
# print(ccode(f.function_def))

# TODO pycode func def needs work
# print(ccode(f.function_def))

# print(pycode(f))
# print(ccode(f))
