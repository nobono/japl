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

# New AeroTable test
# --------------------------------------------------------
from japl.AeroTable.AeroTable import AeroTable
aero = AeroTable("./aerodata/aeromodel_psb.mat")
# print(aero)
quit()

# --------------------------------------------------------
from japl.Util import parse_yaml
from japl import PyQtGraphPlotter
from japl import SimObject
from japl import Model
from japl import Sim
from japl import AeroTable
from japl import Atmosphere
# import mmd
# simobj = mmd.SimObject()

# -------------------------------------------------------------------------
from japl.Library.Vehicles.MissileGenericMMD import (dt,
                                                     state,
                                                     input,
                                                     dynamics,
                                                     static,
                                                     modules,
                                                     defs)
model = Model.from_expression(dt,
                              state,
                              input,
                              dynamics,
                              static_vars=static,
                              modules=modules,
                              definitions=defs,
                              use_multiprocess_build=True)
model.cache_build()
model.set_aerotable()

simobj = SimObject(model)
inits = dict(
        q_0=1,
        q_1=0,
        q_2=0,
        q_3=0,
        r_i_x=6_378_137.0,
        r_i_y=0,
        r_i_z=0,
        v_i_x=50,
        v_i_y=50,
        v_i_z=0,
        alpha=0,
        alpha_dot=0,
        beta=0,
        beta_dot=0,
        p=0,
        wet_mass=100,
        dry_mass=50,

        omega_n=50,
        zeta=0.7,
        K_phi=1,
        omega_p=20,
        phi_c=0,
        T_r=0.5,
        is_boosting=1,
        stage=0,
        is_launched=1)
# -------------------------------------------------------------------------


def input_func(*args):
    return np.array([0.] * 7)


# inits = parse_yaml("./mmd/config_state.yaml")
simobj.init_state(inits)
simobj.set_input_function(input_func)
simobj.plot.set_config({
    "EAST": {"xaxis": "time",
             "yaxis": simobj.r_e},
    "NORTH": {"xaxis": "time",
              "yaxis": simobj.r_n},
    "UP": {"xaxis": "time",
           "yaxis": simobj.r_u},
    })

sim = Sim(t_span=[0, 10], dt=0.1, simobjs=[simobj])

plotter = PyQtGraphPlotter(figsize=[10, 10],
                           frame_rate=30,
                           aspect="auto",
                           axis_color="grey",
                           background_color="black",
                           antialias=False,
                           # quiet=True,
                           )

# plotter.animate(sim).show()
sim.run()

# print(simobj.Y)
