import numpy as np
import japl
from japl.BuildTools.DirectUpdate import DirectUpdateSymbol
from japl import Model
from derivation.nav.code_gen import CCodeGenerator
from sympy import cse, symbols, Matrix


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
