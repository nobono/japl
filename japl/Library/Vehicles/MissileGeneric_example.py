from pprint import pprint
import numpy as np
from sympy import Expr, Matrix, MatrixSymbol, symbols, Piecewise
from sympy import sqrt
from japl import Model
from japl.Math.MathSymbolic import zero_protect_sym
from sympy import Function
from sympy import simplify



class MissileGeneric(Model):
    pass


t = symbols("t")
dt = symbols("dt")

##################################################
# States
##################################################
pos_x, pos_y, pos_z = symbols("pos_x, pos_y, pos_z", cls=Function) #type:ignore
vel_x, vel_y, vel_z = symbols("vel_x, vel_y, vel_z", cls=Function) #type:ignore
w_x, w_y, w_z = symbols("angvel_x, angvel_y, angvel_z", cls=Function) #type:ignore
q_0, q_1, q_2, q_3 = symbols("q_0, q_1, q_2, q_3", cls=Function) #type:ignore

pos = Matrix([pos_x(t), pos_y(t), pos_z(t)])
vel = Matrix([vel_x(t), vel_y(t), vel_z(t)])
w = Matrix([w_x(t), w_y(t), w_z(t)])
q = Matrix([q_0(t), q_1(t), q_2(t), q_3(t)])
mass = symbols("mass")
gravity = Matrix(symbols("gravity_x gravity_y gravity_z"))
speed = symbols("speed")

##################################################
# Inputs
##################################################
acc = Matrix(symbols("acc_x acc_y acc_z"))
torque = Matrix(symbols("torque_x torque_y torque_z"))

wx, wy, wz = w
Sw = Matrix([
    [ 0,   wx,  wy,  wz], #type:ignore
    [-wx,  0,  -wz,  wy], #type:ignore
    [-wy,  wz,   0, -wx], #type:ignore
    [-wz, -wy,  wx,   0], #type:ignore
    ])

pos_new = pos + vel * dt
vel_new = vel + (acc + gravity) * dt
w_new = w + torque * dt
q_new = q + (-0.5 * Sw * q) * dt

##################################################
# Subs for differential definitions
##################################################
diffsub = (
           (pos[0].diff(t), pos_new[0].diff(dt)),    #type:ignore
           (pos[1].diff(t), pos_new[1].diff(dt)),    #type:ignore
           (pos[2].diff(t), pos_new[2].diff(dt)),    #type:ignore
           (vel[0].diff(t), vel_new[0].diff(dt)),    #type:ignore
           (vel[1].diff(t), vel_new[1].diff(dt)),    #type:ignore
           (vel[2].diff(t), vel_new[2].diff(dt)),    #type:ignore
           (w[0].diff(t), w_new[0].diff(dt)),        #type:ignore
           (w[1].diff(t), w_new[1].diff(dt)),        #type:ignore
           (w[2].diff(t), w_new[2].diff(dt)),        #type:ignore
           (q[0].diff(t), q_new[0].diff(dt)),        #type:ignore
           (q[1].diff(t), q_new[1].diff(dt)),        #type:ignore
           (q[2].diff(t), q_new[2].diff(dt)),        #type:ignore
           (q[3].diff(t), q_new[3].diff(dt)),        #type:ignore
           )

##################################################
# Subs for final state vars
##################################################
varsub = {
        pos[0]: symbols("pos_x"),           #type:ignore
        pos[1]: symbols("pos_y"),           #type:ignore
        pos[2]: symbols("pos_z"),           #type:ignore
        vel[0]: symbols("vel_x"),           #type:ignore
        vel[1]: symbols("vel_y"),           #type:ignore
        vel[2]: symbols("vel_z"),           #type:ignore
        w[0]: symbols("angvel_x"),          #type:ignore
        w[1]: symbols("angvel_y"),          #type:ignore
        w[2]: symbols("angvel_z"),          #type:ignore
        q[0]: symbols("q_0"),               #type:ignore
        q[1]: symbols("q_1"),               #type:ignore
        q[2]: symbols("q_2"),               #type:ignore
        q[3]: symbols("q_3"),               #type:ignore
        torque[0]: symbols("torque_x"),     #type:ignore
        torque[1]: symbols("torque_y"),     #type:ignore
        torque[2]: symbols("torque_z"),     #type:ignore
        }

speed_new = (vel.dot(vel))**0.5

##################################################
# Define State Update
##################################################
X_new = Matrix([
    pos,
    vel,
    w_new.as_mutable(),
    q_new.as_mutable(),
    mass,
    gravity,
    speed_new,
    ])

state = Matrix([pos, vel, w, q, mass, gravity, speed])
input = Matrix([acc, torque])


##################################################
# Issues observed passing subs as tuple of tuples.
# ensure the subs argument is formated as dict.
##################################################
# diffsub = ((pos.diff(t), pos_new.diff(dt)), )
diffsub_dict = {}
if isinstance(diffsub, tuple) or isinstance(diffsub, list):
    # update a new dict with substition pairs.
    # if pair is Matrix or MatrixSymbol (N x 1),
    # update each element.
    for old, new in diffsub:
        if hasattr(old, "__len__") and hasattr(new, "__len__"):
            for elem_old, elem_new in zip(old, new): #type:ignore
                diffsub_dict[elem_old] = elem_new
        else:
            try:
                diffsub_dict[old] = new
            except Exception as e:
                raise Exception(e, "\nunhandled case. old and new need to both have '__len__'.")
else:
    diffsub_dict = diffsub

##################################################
# Define dynamics
##################################################
dynamics = Matrix(X_new.diff(t).subs(diffsub_dict).doit())
dynamics = dynamics.subs(varsub)


model = MissileGeneric().from_expression(dt, state, input, dynamics)


##################################################
# calculate dynamics manually & add to array
# speed_dot = simplify(norm.diff(dt).subs(varsub))
##################################################

# vel_norm = ((vel.T * vel)**0.5)[0]
# speed_dot = vel.dot((acc + gravity)) / vel_norm

# add to dynamics directly
# dynamics = Matrix([dynamics,
#                    speed_dot])

##################################################
# from_function example
##################################################

# A = dynamics.jacobian(state) #type:ignore
# B = dynamics.jacobian(input) #type:ignore
# from japl.Math.Rotation import Sw as Sw_
# def func(X, U, dt, *args):
#     pos = X[:3]
#     vel = X[3:6]
#     w = X[6:9]
#     q = X[9:13]
#     mass = X[13]
#     gravity = X[14:17]
#     speed = X[17]

#     acc = U[:3]
#     tq = U[3:6]

#     x_new = vel
#     v_new = (acc + gravity)
#     w_new = tq
#     q_new = (-0.5 * Sw_(w) @ q)
#     mass_new = 0
#     gravity_new = np.zeros(3)
#     # speed_new = (np.sqrt(vel.T @ vel) - speed)
#     # speed_new = (np.linalg.norm(vel) - speed)
#     speed_new = ((vel[0]*(acc[0] + gravity[0]) + vel[1]*(acc[1] + gravity[1]) + vel[2]*(acc[2] + gravity[2])) / np.linalg.norm(vel))

#     Xdot = np.array([*x_new, *v_new, *w_new, *q_new, mass_new, *gravity_new, speed_new])
#     return Xdot

# model = RigidBodyModel().from_function(dt, state, input, func)

