from sympy import MatrixSymbol
from japl.BuildTools import CodeGeneration
from japl.Library.Vehicles.RigidBodyModel import model, state, input



codegen = CodeGeneration.CCodeGenerator()
codegen.write_function_to_file(path="./temp_dev/testCode.cpp",
                               function_name="test_func",
                               expr=model.dynamics_expr,
                               input_vars=[state, input],
                               return_var=MatrixSymbol("Xnew", *model.dynamics_expr.shape))  # type:ignore

# codegen = CodeGeneration.OctaveCodeGenerator()
# codegen.write_function_to_file(path="./testCode.m",
#                                function_name="test_func",
#                                expr=model.dynamics_expr,
#                                input_vars=[state, input],
#                                return_var=MatrixSymbol("Xnew", *model.dynamics_expr.shape))


# codegen = CodeGeneration.PyCodeGenerator()
# codegen.write_function_to_file(path="./testCode.py",
#                                function_name="test_func",
#                                expr=model.dynamics_expr,
#                                input_vars=[state, input],
#                                return_var=MatrixSymbol("Xnew", *model.dynamics_expr.shape))
