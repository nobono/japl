from sympy import symbols
from sympy import Function
from sympy import Matrix
from sympy import Symbol
from japl import JaplFunction
from japl.CodeGen import ccode, pycode



t = Symbol("t")
dt = Symbol("dt")

pos = Symbol("pos")
vel = Symbol("vel")
acc = Symbol("acc")

pos_new = pos + vel * dt
vel_new = vel + acc * dt
state_new = Matrix([pos_new, vel_new])

state = Matrix([pos, vel])
input = Matrix([acc])
static = Matrix([])

# X = Matrix([pos_new, vel_new])
# X_dot = X.diff(dt)


class linear_motion(JaplFunction):
    expr = state_new


func = linear_motion(acc)
func._build("c")
# print(ccode(func))
print(ccode(func.function_def))

# class dynamics(JaplFunction):
#     expr = X_dot


# f = dynamics(X)

# pos_new = pos + vel * dt + (0.5 * acc) * dt**2
# print(pos_new.diff(dt))
# print(pos_new.diff(dt, 2))
