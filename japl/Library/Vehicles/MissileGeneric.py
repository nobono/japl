from pprint import pprint
import numpy as np
from sympy import Matrix, MatrixSymbol, symbols, Piecewise
from sympy import sqrt
from japl import Model
from sympy import Function



class RigidBodyModel(Model):
    pass


dt = symbols("dt")

# states
pos = Matrix(symbols("pos_x pos_y pos_z"))      # must be fixed for AeroModel
vel = Matrix(symbols("vel_x vel_y vel_z"))   # must be fixed for AeroModel
w = Matrix(symbols("angvel_x angvel_y angvel_z"))
q = Matrix(symbols("q_0 q_1 q_2 q_3"))  # must be fixed for AeroModel
mass = symbols("mass")
gravity = Matrix(symbols("gravity_x gravity_y gravity_z"))

# inputs
acc = Matrix(symbols("acc_x acc_y acc_z"))
tq = Matrix(symbols("torque_x torque_y torque_z"))

# define state update
w_skew = Matrix(w).hat()
Sw = Matrix(np.zeros((4,4)))
Sw[0, :] = Matrix([0, *w]).T
Sw[:, 0] = Matrix([0, *-w])
Sw[1:, 1:] = w_skew

x_new = pos + vel * dt
v_new = vel + (acc + gravity) * dt
w_new = w + tq * dt
q_new = q + (-0.5 * Sw * q) * dt

X_new = Matrix([
    x_new.as_mutable(),
    v_new.as_mutable(),
    w_new.as_mutable(),
    q_new.as_mutable(),
    mass,
    gravity,
    ])

state = Matrix([pos, vel, w, q, mass, gravity])
input = Matrix([acc, tq])

dynamics = X_new.diff(dt)

# A = dynamics.jacobian(state) #type:ignore
# B = dynamics.jacobian(input) #type:ignore

model = RigidBodyModel().from_expression(dt, state, input, dynamics)

