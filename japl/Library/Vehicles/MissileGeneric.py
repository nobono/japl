from pprint import pprint
import numpy as np
from sympy import Matrix, MatrixSymbol, symbols, Piecewise
from sympy import sqrt
from japl import Model
from sympy import Function
from sympy import simplify



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
speed = symbols("speed")

# inputs
acc = Matrix(symbols("acc_x acc_y acc_z"))
tq = Matrix(symbols("torque_x torque_y torque_z"))

wx, wy, wz = w
Sw = Matrix([
    [ 0,   wx,  wy,  wz], #type:ignore
    [-wx,  0,  -wz,  wy], #type:ignore
    [-wy,  wz,   0, -wx], #type:ignore
    [-wz, -wy,  wx,   0], #type:ignore
    ])

x_new = pos + vel * dt
v_new = vel + (acc + gravity) * dt
w_new = w + tq * dt
q_new = q + (-0.5 * Sw * q) * dt

###########################################
# subs = {vel[0]: 1500,       #type:ignore
#         vel[1]: 1,       #type:ignore
#         vel[2]: 1,       #type:ignore
#         acc[0]: 0,       #type:ignore
#         acc[1]: 0,       #type:ignore
#         acc[2]: -.1,       #type:ignore
#         gravity[0]: 0,       #type:ignore
#         gravity[1]: 0,       #type:ignore
#         gravity[2]: -9.8,       #type:ignore
#         dt: 0.01,       #type:ignore
#         }

# fpos_x, fpos_y, fpos_z = symbols("fpos_x fpos_y fpos_z", cls=Function) #type:ignore
# fpos = Matrix([fpos_x(dt), fpos_y(dt), fpos_z(dt)])

# another way
# fvel_x = fpos_x(dt).diff(dt)
# fvel_y = fpos_y(dt).diff(dt)
# fvel_z = fpos_z(dt).diff(dt)
# fvel = Matrix([fvel_x, fvel_y, fvel_z])

# facc_x = fpos_x(dt).diff(dt, 2)
# facc_y = fpos_y(dt).diff(dt, 2)
# facc_z = fpos_z(dt).diff(dt, 2)
# facc = Matrix([facc_x, facc_y, facc_z])

# fvel = Matrix(fpos.diff(dt))
# facc = Matrix(fvel.diff(dt))

# diffsub = (
#            # *[(old, new) for old, new in zip(fvel, vel)], #type:ignore
#            # *[(old, new) for old, new in zip(facc, v_new.diff(dt))], #type:ignore

#            (fvel[0], vel[0]),
#            (fvel[1], vel[1]),
#            (fvel[2], vel[2]),
#            (facc[0], v_new.diff(dt)[0]),
#            (facc[1], v_new.diff(dt)[1]),
#            (facc[2], v_new.diff(dt)[2]),

#            # another way
#            # fvel_x: vel[0],
#            # fvel_y: vel[1],
#            # fvel_z: vel[2],
#            # facc_x: v_new.diff(dt)[0],
#            # facc_y: v_new.diff(dt)[1],
#            # facc_z: v_new.diff(dt)[2],
#            )

# speed_new = (fvel.dot(fvel))**0.5

X_new = Matrix([
    x_new.as_mutable(),
    v_new.as_mutable(),
    w_new.as_mutable(),
    q_new.as_mutable(),
    mass,
    gravity,
    # speed_new,
    ])

state = Matrix([pos, vel, w, q, mass, gravity, speed])
input = Matrix([acc, tq])

# if isinstance(diffsub, tuple) or isinstance(diffsub, list):
#     diffsub_dict = {k: v for k, v in diffsub}
# else:
#     diffsub_dict = diffsub

vel_norm = ((vel.T * vel)**0.5)[0]
speed_dot = vel.dot((acc + gravity)) / vel_norm

dynamics = Matrix(X_new.diff(dt))
dynamics = Matrix([
    dynamics,
    speed_dot
    ])


model = RigidBodyModel().from_expression(dt, state, input, dynamics)

