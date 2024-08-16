import time
from japl.Util.Desym import Desym
# from sympy.utilities.autowrap import autowrap
# from sympy.utilities.codegen import codegen
# from sympy.utilities.codegen import C99CodeGen, CodeGen, Routine
from sympy import Matrix, symbols
# from sympy import lambdify
import numpy as np
from libraries.Custom.wrapper_module_0 import autofunc_c as mod  # type:ignore
from numba import njit



# Model
####################################
pos = Matrix(symbols("x y z"))      # must be fixed for AeroModel
vel = Matrix(symbols("vx vy vz"))   # must be fixed for AeroModel
acc = Matrix(symbols("ax ay az"))
tq = Matrix(symbols("tqx tqy tqz"))
w = Matrix(symbols("wx wy wz"))
q = Matrix(symbols("q0 q1 q2 q3"))  # must be fixed for AeroModel

dt = symbols("dt")
mass = symbols("mass")

w_skew = Matrix(w).hat()         # type:ignore
Sw = Matrix(np.zeros((4, 4)))
Sw[0, :] = Matrix([0, *w]).T
Sw[:, 0] = Matrix([0, *-w])      # type:ignore
Sw[1:, 1:] = w_skew

x_new = pos + vel * dt
v_new = vel + acc * dt
w_new = w + tq * dt
q_new = q + (-0.5 * Sw * q) * dt
mass_new = mass

X_new = Matrix([
    x_new.as_mutable(),
    v_new.as_mutable(),
    w_new.as_mutable(),
    q_new.as_mutable(),
    mass_new,
    ])

state = Matrix([pos, vel, w, q, mass])
input = Matrix([acc, tq])

dynamics: Matrix = X_new.diff(dt)  # type:ignore


lamf = Desym((state, input, dt), dynamics)
# lamf = lambdify((state, input, dt), dynamics)

outdir = "japl/Library/Custom"

# # Generate C code for the expression
# [(c_name, c_code), (h_name, c_header)] = codegen(
#     name_expr=("func", dynamics),
#     language="C",
#     prefix="func",
#     project="myproject",
#     header=True,
#     empty=False,
#     # code_gen=C99CodeGen(),
#     # to_files=
#     )
# with open(f"{outdir}/{c_name}", "a+") as f:
#     f.write(c_code)
# with open(f"{outdir}/{h_name}", "a+") as f:
#     f.write(c_header)

# Create a custom C99CodeGen instance
# class CustomCodeGen(C99CodeGen):
#     def get_module_name(self, prefix):
#         return "mymodule"
# code_gen = CustomCodeGen()
# Routine("myfunc", )

# wrapf = autowrap(expr=dynamics,
#                  args=(*state, *input, dt),
#                  language="C",
#                  include_dirs=[],
#                  # library_dirs=
#                  # libraries=,
#                  verbose=True,
#                  backend="cython",
#                  tempdir=outdir,
#                  # code_gen=code_gen,
#                  )


pm = ([0,0,0, 0,0,0, 0,0,0, 1,0,0,0, 10], [1,0,0, 0,0,0], 0.01)     # noqa


# @njit
def pyf(X, U, dt):
    wx = X[6]
    wy = X[7]
    wz = X[8]
    q0 = X[9]
    q1 = X[10]
    q2 = X[11]
    q3 = X[12]
    return np.array([
        X[3],
        X[4],
        X[5],
        U[0],
        U[1],
        U[2],
        U[3],
        U[4],
        U[5],
        -0.5*q1*wx - 0.5*q2*wy - 0.5*q3*wz,     # noqa
        0.5*q0*wx + 0.5*q2*wz - 0.5*q3*wy,     # noqa
        0.5*q0*wy - 0.5*q1*wz + 0.5*q3*wx,     # noqa
        0.5*q0*wz + 0.5*q1*wy - 0.5*q2*wx,     # noqa
        0,
        ])


N = 1_000_000
st = time.time()
for i in range(N):
    pyf(*pm)
py_time = time.time() - st
print("\nexec py:", py_time)

st = time.time()
for i in range(N):
    lamf(*pm)
lam_time = time.time() - st
print("\nexec lam:", lam_time)

st = time.time()
for i in range(N):
    # wrapf(*pm[0], *pm[1], pm[2])
    mod(*pm[0], *pm[1], pm[2])
wrap_time = time.time() - st
print("\nexec wrap:", wrap_time)


print(f"\n{py_time / wrap_time}")
