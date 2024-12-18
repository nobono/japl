from sympy import symbols
from sympy import Function
from sympy import Matrix
from sympy import Symbol
from japl.CodeGen import JaplFunction
from japl.CodeGen import ccode, pycode
from japl import Model
from japl import SimObject
from japl.CodeGen.Ast import JaplClass
from japl.CodeGen import FileBuilder



t = symbols("t")
dt = symbols("dt")

pos_x = Symbol("pos_x")
pos_y = Symbol("pos_y")
pos_z = Symbol("pos_z")

vel_x = Symbol("vel_x")
vel_y = Symbol("vel_y")
vel_z = Symbol("vel_z")

acc_x = Symbol("acc_x")
acc_y = Symbol("acc_y")
acc_z = Symbol("acc_z")

pos = Matrix([pos_x, pos_y, pos_z])
vel = Matrix([vel_x, vel_y, vel_z])
acc = Matrix([acc_x, acc_y, acc_z])

pos_new = pos + vel * dt
vel_new = vel + acc * dt

state = Matrix([pos, vel])
input = Matrix([acc])
static = Matrix([])

X = Matrix([pos_new, vel_new])
X_dot = X.diff(dt)


class dynamics(JaplFunction):
    expr = X_dot


f = dynamics(X)
f._build_function("c")
# print(pycode(f.function_def))

model = Model.from_expression(dt, state, input, X_dot)
# simobj = SimObject(model)
# simobj.init_state([0] * len(state))

model.create_c_module("mtest")

# cl = JaplClass("myClass", parent="Model", members={"state vars": state,
#                                                    "input vars": input,
#                                                    "static vars": static})
# print(pycode(cl))

# fb = FileBuilder("stubs.pyi", contents=pycode(cl))
# fb.build()
# fb.dumps(path="./")
