import numpy as np
from sympy import Matrix, symbols
from japl import Model



class RigidBodyModel(Model):
    pass


dt = symbols("dt")

# states
pos = Matrix(symbols("x y z"))      # must be fixed for AeroModel
vel = Matrix(symbols("vx vy vz"))   # must be fixed for AeroModel
w = Matrix(symbols("wx wy wz"))
q = Matrix(symbols("q0 q1 q2 q3"))  # must be fixed for AeroModel
mass = symbols("mass")
gravity = Matrix(symbols("gravity_x gravity_y gravity_z"))

# inputs
acc = Matrix(symbols("ax ay az"))
tq = Matrix(symbols("tqx tqy tqz"))

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
# mass_new = mass
# gravity_new = gravity

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

model = RigidBodyModel().from_expression(dt, state, input, dynamics)
