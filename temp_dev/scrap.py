import numpy as np
import japl
from japl.BuildTools.DirectUpdate import DirectUpdateSymbol
from japl import Model
# from derivation.nav.code_gen import CCodeGenerator
from sympy import cse, symbols, Matrix, MatrixSymbol
from japl.Util.Desym import Desym
from time import perf_counter
from sympy.utilities.autowrap import autowrap


"""simple setup"""
from sympy.abc import x, y, z, a, b, c
import sympy as sp

expr = sp.sqrt(x)**4 + sp.sqrt(y)**2 + sp.sin(z)**3 + sp.cos(a + b + c)

# desym_wrap = autowrap(expr=expr, args=(a,b,c, x,y,z), language="C", backend="cython")


def desym_py(a, b, c, x, y, z):
    return np.sqrt(x)**4 + np.sqrt(y)**2 + np.sin(z)**3 + np.cos(a + b + c)


desym_wrap = Desym((a, b, c, x, y, z), expr, wrap_type="autowrap")
desym_lamb = Desym((a, b, c, x, y, z), expr, wrap_type="lambdify")

N = 1_000_000
st = perf_counter()
for i in range(N):
    ret = desym_lamb(1, 2, 3, 4, 5, 6)
print("exec:", perf_counter() - st)

st = perf_counter()
for i in range(N):
    ret = desym_wrap(1, 2, 3, 4, 5, 6)
print("exec:", perf_counter() - st)

st = perf_counter()
for i in range(N):
    ret = desym_py(1, 2, 3, 4, 5, 6)
print("exec:", perf_counter() - st)

quit()

model = japl.Model.from_file(f"{japl.JAPL_HOME_DIR}/data/mmd.japl")
d, s, i = model.dump_code()


def unpack_vars(vars):
    ret = []
    for var in vars:
        if hasattr(var, "__len__"):
            ret += unpack_vars(var)
        else:
            ret += [var]
    return ret


def write_function_to_file(path: str, model: Model, codegen):
    dynamics_simple = cse(model.dynamics_expr, symbols("X0:1000"), optimizations="basic")
    vars = unpack_vars(model.vars)
    gen = codegen(path)
    gen.write_function_definition(name="dynamics", args=vars)
    gen.write_subexpressions(dynamics_simple[0])
    gen.write_matrix(matrix=Matrix(dynamics_simple[1]),
                     variable_name="Xdot")
    gen.close()


path = f"{japl.JAPL_HOME_DIR}/data/test_code.cpp"
write_function_to_file(path, model, CCodeGenerator)
