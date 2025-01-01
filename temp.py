import numpy as np
# from typing import Any, Callable
# from sympy import Basic
# from sympy import symbols
# from sympy import Matrix
# from sympy import true, false
# from sympy import cse
# from sympy import MatrixSymbol
# from sympy import Symbol
# from sympy import Function
# from sympy import Expr
# from sympy.codegen.ast import CodeBlock, Element
# from sympy.codegen.ast import Type
# from sympy.codegen.ast import String
# from sympy.codegen.ast import Token
# from sympy.codegen.ast import Tuple
# from japl.CodeGen import pycode
# from japl.CodeGen import ccode
# from japl.CodeGen import Builder
# # from japl.CodeGen import FileBuilder
# # from japl.CodeGen import CFileBuilder
# # from japl.CodeGen import ModuleBuilder
# # from japl.CodeGen import CodeGenerator
# # from japl.CodeGen import JaplFunction
# # from japl.CodeGen.Ast import CTypes, JaplType, JaplTypes, Kwargs, CType, PyType
# # from japl.CodeGen.Ast import convert_symbols_to_variables
# # from japl.CodeGen.Ast import Dict
# # from japl.CodeGen.JaplFunction import numbered_symbols, get_lang_types, Variable
# # from japl.CodeGen.Util import optimize_expression
# # from japl.Library.Earth.Earth import Earth
# # from sympy import sin, cos
# # from japl.CodeGen.Ast import JaplClass
# from japl import SimObject
# from japl.CodeGen.Ast import CodeGenFunctionCall
# from japl.CodeGen.Ast import Kwargs
# from japl.CodeGen.JaplFunction import JaplFunction
# from japl.Util.Util import iter_type_check
# from pprint import pprint



# import mmd
# m = mmd.Model()
# s = mmd.SimObject()
# print(m.state_dim)
# print(s.state_dim)


from japl import PyQtGraphPlotter
from japl import Sim
import lin


plotter = PyQtGraphPlotter(frame_rate=30,
                           aspect="auto",
                           axis_color="grey",
                           background_color="black",
                           antialias=False,
                           # quiet=True,
                           )


def input_func(*args):
    return np.array([1.])


simobj = lin.SimObject()
simobj.init_state([0, 0])
simobj.set_input_function(input_func)
simobj.plot.set_config({
    "XY": {"xaxis": "time",
           "yaxis": simobj.pos}
    })

sim = Sim(t_span=[0, 10], dt=0.1, simobjs=[simobj])
# sim.run()
plotter.animate(sim).show()

# print(simobj.Y)
