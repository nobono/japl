from sympy import symbols
from sympy import Function
from sympy import Matrix
from sympy import Symbol
from japl.CodeGen import JaplFunction
from japl.CodeGen import ccode, pycode



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

X = Matrix([pos_new, vel_new])
X_dot = X.diff(dt)


class dynamics(JaplFunction):
    expr = X_dot


f = dynamics(X)
f._build_function("c")
print()
print(ccode(f.function_def))
# print(X_dot)
