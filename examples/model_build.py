from pprint import pprint
import sympy as sp
from sympy import symbols, Matrix, MatrixSymbol
import numpy as np


dt = symbols("dt")
x = MatrixSymbol('x', 3, 1)
v = MatrixSymbol('v', 3, 1)
a = MatrixSymbol('a', 3, 1)
tq = MatrixSymbol('tq', 3, 1)
w = MatrixSymbol('w', 3, 1)
q = MatrixSymbol('q', 4, 1)
mass = symbols("mass")

Sq = Matrix([
    [-q[1], -q[2], -q[3]],
    [q[0], -q[3], q[2]],
    [q[3], q[0], -q[1]],
    [-q[2], q[1], q[0]],
    ])

w_skew = Matrix(w).hat()    # type:ignore
Sw = Matrix(np.zeros((4, 4)))
Sw[0, :] = Matrix([0, *w]).T
Sw[:, 0] = Matrix([0, *-w])     # type:ignore
Sw[1:, 1:] = w_skew


x_new = x + v * dt
v_new = v + a * dt
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


X = Matrix([x, v, w, q, mass])
U = Matrix([a, tq])


dynamics: Matrix = X_new.diff(dt)   # type:ignore

A = dynamics.jacobian(X)
B = dynamics.jacobian(U)

X_dot = A * X + B * U

dyn_func = sp.lambdify((X, U), X_dot, cse=True)

pprint(A)
print()
pprint(B)
print()
pprint(X_dot)
print()
pprint(dyn_func(
    [0, 0, 0,
     1, 0, 0,
     0, 0, 0,
     1, 0, 0, 0],
    [0, 1, 0,
     0, 0, 0]))

pass
