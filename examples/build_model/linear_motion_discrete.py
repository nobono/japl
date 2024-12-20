from sympy import Matrix
from sympy import Symbol
from japl import JaplFunction
from japl import Model
from japl.CodeGen import ccode, octave_code


# define variables
t = Symbol("t")
dt = Symbol("dt")
pos = Symbol("pos")
vel = Symbol("vel")
acc = Symbol("acc")

state = Matrix([pos, vel])
input = Matrix([acc])

# define linear motion dynamics
pos_new = pos + vel * dt
vel_new = vel + acc * dt
state_new = Matrix([pos_new, vel_new])
dynamics = state_new.diff(dt)

model = Model.from_expression(dt_var=dt,
                              state_vars=state,
                              input_vars=input,
                              dynamics_expr=dynamics)

model.create_c_module("lin")
