from sympy import Matrix
from sympy import Symbol
from japl import JaplFunction
from japl.CodeGen import ccode, octave_code



# define variables
t = Symbol("t")
dt = Symbol("dt")
pos = Symbol("pos")
vel = Symbol("vel")
acc = Symbol("acc")

# define linear motion dynamics
pos_new = pos + vel * dt
vel_new = vel + acc * dt
state_new = Matrix([pos_new, vel_new])

state = Matrix([pos, vel])
input = Matrix([acc])


# create JaplFunction subclass
class linear_motion(JaplFunction):
    expr = state_new


func = linear_motion(pos, vel, acc, dt)

# generate code to a target language
ccode_str = ccode(func._build("c"))
octave_code_str = octave_code(func._build("octave"))

print("c++ code generation output:")
print(ccode_str)
print()
print("octave / matlab code generation output:")
print(octave_code_str)
