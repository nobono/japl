from sympy import Matrix, Symbol, symbols
from sympy import sqrt
from japl import Model
from japl.Math.MathSymbolic import zero_protect_sym
from sympy import Function

from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic
from japl.BuildTools.DirectUpdate import DirectUpdate



class MissileGeneric(Model):
    pass


t = symbols("t")
dt = symbols("dt")

pos_x = Function("pos_x")(t) #type:ignore
pos_y = Function("pos_y")(t) #type:ignore
pos_z = Function("pos_z")(t) #type:ignore

vel_x = Function("vel_x")(t) #type:ignore
vel_y = Function("vel_y")(t) #type:ignore
vel_z = Function("vel_z")(t) #type:ignore

angvel_x = Function("angvel_x")(t) #type:ignore
angvel_y = Function("angvel_y")(t) #type:ignore
angvel_z = Function("angvel_z")(t) #type:ignore

q_0 = Function("q_0")(t) #type:ignore
q_1 = Function("q_1")(t) #type:ignore
q_2 = Function("q_2")(t) #type:ignore
q_3 = Function("q_3")(t) #type:ignore

gravity_x = Function("gravity_x")(t) #type:ignore
gravity_y = Function("gravity_y")(t) #type:ignore
gravity_z = Function("gravity_z")(t) #type:ignore

acc_x = Function("acc_x")(t) #type:ignore
acc_y = Function("acc_y")(t) #type:ignore
acc_z = Function("acc_z")(t) #type:ignore

torque_x = Function("torque_x")(t) #type:ignore
torque_y = Function("torque_y")(t) #type:ignore
torque_z = Function("torque_z")(t) #type:ignore

##################################################
# States
##################################################

pos = Matrix([pos_x, pos_y, pos_z])
vel = Matrix([vel_x, vel_y, vel_z])
angvel = Matrix([angvel_x, angvel_y, angvel_z])
q = Matrix([q_0, q_1, q_2, q_3])
mass = symbols("mass")
gravity = Matrix([gravity_x, gravity_y, gravity_z])
speed = symbols("speed", cls=Function)(t) #type:ignore

##################################################
# Inputs
##################################################

acc = Matrix([acc_x, acc_y, acc_z])
torque = Matrix([torque_x, torque_y, torque_z])

##################################################
# Update Equations
##################################################

wx, wy, wz = angvel
Sw = Matrix([
    [ 0,   wx,  wy,  wz], #type:ignore
    [-wx,  0,  -wz,  wy], #type:ignore
    [-wy,  wz,   0, -wx], #type:ignore
    [-wz, -wy,  wx,   0], #type:ignore
    ])

pos_new = pos + vel * dt
vel_new = vel + (acc + gravity) * dt
w_new = angvel + torque * dt
q_new = q + (-0.5 * Sw * q) * dt
mass_new = mass

pos_dot = pos_new.diff(dt)
vel_dot = vel_new.diff(dt)
angvel_dot = w_new.diff(dt)
q_dot = q_new.diff(dt)
mass_dot = mass_new.diff(dt)

##################################################
# Equations for Aerotable / Atmosphere
##################################################

# gravity
atmosphere = AtmosphereSymbolic()
gravity_new = Matrix([0, 0, -atmosphere.grav_accel(pos_z)]) #type:ignore

# speed
speed_new = sqrt(vel.dot(vel))
speed_dot = speed_new.diff(t)

##################################################
# Differential Definitions
##################################################

defs = (
        (pos.diff(t),       pos_dot),
        (vel.diff(t),       vel_dot),
        (angvel.diff(t),    angvel_dot),
        (q.diff(t),         q_dot),
        (mass.diff(t),      mass_dot),
        (speed.diff(t),     speed_dot),
        )

##################################################
# Define State & Input Arrays
##################################################

state = Matrix([
    pos,
    vel,
    angvel,
    q,
    mass,
    DirectUpdate(gravity, gravity_new),
    speed,
    ])

input = Matrix([acc, torque])

##################################################
# Define dynamics
##################################################

dynamics = Matrix(state.diff(t))

##################################################
# Build Model
##################################################

model = MissileGeneric().from_expression(dt, state, input, dynamics,
                                         modules=atmosphere.modules,
                                         definitions=defs)

