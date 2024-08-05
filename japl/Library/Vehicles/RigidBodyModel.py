import numpy as np
from sympy import Matrix, MatrixSymbol, symbols, Piecewise, Function
from japl import Model

from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.Model.BuildTools.DirectUpdate import DirectUpdate



class RigidBodyModel(Model):
    pass


t = symbols("t")
dt = symbols("dt")

# states
pos_x, pos_y, pos_z = symbols("pos_x, pos_y, pos_z", cls=Function) #type:ignore
vel_x, vel_y, vel_z = symbols("vel_x, vel_y, vel_z", cls=Function) #type:ignore
w_x, w_y, w_z = symbols("angvel_x, angvel_y, angvel_z", cls=Function) #type:ignore
q_0, q_1, q_2, q_3 = symbols("q_0, q_1, q_2, q_3", cls=Function) #type:ignore
gravity_x, gravity_y, gravity_z = symbols("gravity_x, gravity_y, gravity_z") #type:ignore

pos = Matrix([pos_x(t), pos_y(t), pos_z(t)])
vel = Matrix([vel_x(t), vel_y(t), vel_z(t)])
angvel = Matrix([w_x(t), w_y(t), w_z(t)])
q = Matrix([q_0(t), q_1(t), q_2(t), q_3(t)])
mass = symbols("mass")
gravity = Matrix([gravity_x, gravity_y, gravity_z])
speed = symbols("speed", cls=Function)(t) #type:ignore

# inputs
acc = Matrix(symbols("acc_x acc_y acc_z"))
tq = Matrix(symbols("torque_x torque_y torque_z"))

# define state update
wx, wy, wz = angvel
Sw = Matrix([
    [ 0,   wx,  wy,  wz], #type:ignore
    [-wx,  0,  -wz,  wy], #type:ignore
    [-wy,  wz,   0, -wx], #type:ignore
    [-wz, -wy,  wx,   0], #type:ignore
    ])

pos_new = pos + vel * dt
vel_new = vel + (acc + gravity) * dt
w_new = angvel + tq * dt
q_new = q + (-0.5 * Sw * q) * dt
mass_new = mass

pos_dot = pos_new.diff(dt)
vel_dot = vel_new.diff(dt)
w_dot = w_new.diff(dt)
q_dot = q_new.diff(dt)
mass_dot = mass_new.diff(dt)

atmosphere = AtmosphereSymbolic()
gravity_new = Matrix([0, 0, -atmosphere.grav_accel(pos_z(t))]) #type:ignore

defs = (
        (pos.diff(t),       pos_dot),
        (vel.diff(t),       vel_dot),
        (angvel.diff(t),    w_dot),
        (q.diff(t),         q_dot),
        )

state = Matrix([
    pos,
    vel,
    angvel,
    q,
    mass,
    DirectUpdate(gravity, gravity_new),
    ])

input = Matrix([acc, tq])

dynamics = state.diff(t)

model = RigidBodyModel().from_expression(dt, state, input, dynamics,
                                         definitions=defs,
                                         modules=atmosphere.modules)
