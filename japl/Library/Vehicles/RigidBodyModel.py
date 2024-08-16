from sympy import Matrix, symbols, Symbol
from sympy.core.function import Function
from japl import Model
from japl.BuildTools.DirectUpdate import DirectUpdate



class RigidBodyModel(Model):
    pass


t = symbols("t")
dt = symbols("dt")

pos_x = Function("pos_x", real=True)(t) #type:ignore
pos_y = Function("pos_y", real=True)(t) #type:ignore
pos_z = Function("pos_z", real=True)(t) #type:ignore

vel_x = Function("vel_x", real=True)(t) #type:ignore
vel_y = Function("vel_y", real=True)(t) #type:ignore
vel_z = Function("vel_z", real=True)(t) #type:ignore

angvel_x = Function("angvel_x", real=True)(t) #type:ignore
angvel_y = Function("angvel_y", real=True)(t) #type:ignore
angvel_z = Function("angvel_z", real=True)(t) #type:ignore

q_0 = Function("q_0", real=True)(t) #type:ignore
q_1 = Function("q_1", real=True)(t) #type:ignore
q_2 = Function("q_2", real=True)(t) #type:ignore
q_3 = Function("q_3", real=True)(t) #type:ignore

acc_x = Function("acc_x", real=True)(t) #type:ignore
acc_y = Function("acc_y", real=True)(t) #type:ignore
acc_z = Function("acc_z", real=True)(t) #type:ignore

torque_x = Function("torque_x", real=True)(t) #type:ignore
torque_y = Function("torque_y", real=True)(t) #type:ignore
torque_z = Function("torque_z", real=True)(t) #type:ignore

force_x = Function("force_x", real=True)(t) #type:ignore
force_y = Function("force_y", real=True)(t) #type:ignore
force_z = Function("force_z", real=True)(t) #type:ignore

Ixx = Symbol("Ixx", real=True)
Iyy = Symbol("Iyy", real=True)
Izz = Symbol("Izz", real=True)

inertia = Matrix([
    [Ixx, 0, 0],
    [0, Iyy, 0],
    [0, 0, Izz],
    ])

gacc = Symbol("gacc", real=True)

##################################################
# States
##################################################

pos = Matrix([pos_x, pos_y, pos_z])
vel = Matrix([vel_x, vel_y, vel_z])
angvel = Matrix([angvel_x, angvel_y, angvel_z])
quat = Matrix([q_0, q_1, q_2, q_3])
mass = symbols("mass")

##################################################
# Inputs
##################################################

torque = Matrix([torque_x, torque_y, torque_z])
force = Matrix([force_x, force_y, force_z])

##################################################
# Update Equations
##################################################

acc = force / mass
angacc = inertia.inv() * torque
gravity = Matrix([0, 0, gacc])

wx, wy, wz = angvel
Sw = Matrix([
    [ 0,   wx,  wy,  wz], #type:ignore
    [-wx,  0,  -wz,  wy], #type:ignore
    [-wy,  wz,   0, -wx], #type:ignore
    [-wz, -wy,  wx,   0], #type:ignore
    ])

pos_new = pos + vel * dt
vel_new = vel + (acc + gravity) * dt
angvel_new = angvel + angacc * dt
quat_new = quat + (-0.5 * Sw * quat) * dt
mass_new = mass

pos_dot = pos_new.diff(dt)
vel_dot = vel_new.diff(dt)
angvel_dot = angvel_new.diff(dt)
quat_dot = quat_new.diff(dt)
mass_dot = mass_new.diff(dt)

##################################################
# Differential Definitions
##################################################

defs = (
        (pos.diff(t),       pos_dot),
        (vel.diff(t),       vel_dot),
        (angvel.diff(t),    angvel_dot),
        (quat.diff(t),      quat_dot),
        )

##################################################
# Define State & Input Arrays
##################################################

state = Matrix([
    pos,
    vel,
    angvel,
    quat,
    mass,
    Ixx,
    Iyy,
    Izz,
    gacc,
    ])

input = Matrix([
    force,
    torque,
    ])

##################################################
# Define dynamics
##################################################

dynamics = state.diff(t)

##################################################
# Build Model
##################################################

model = RigidBodyModel.from_expression(dt, state, input, dynamics,
                                         definitions=defs,
                                         modules=[])

