from pprint import pprint
import numpy as np
from sympy import Expr, Matrix, MatrixSymbol, Symbol, symbols, Piecewise
from sympy import sqrt
from sympy.matrices.expressions.matexpr import MatrixElement
from japl import Model
from japl.Math.MathSymbolic import zero_protect_sym
from sympy import Function
from sympy import simplify
from japl import StateRegister

from japl.Aero.AtmosphereSymbolic import AtmosphereSymbolic



def process_subs(sub: tuple|list|dict) -> dict:
    """This method is used to convert differntial definitions
    into a substitutable dict."""
    ret = {}
    if isinstance(sub, tuple) or isinstance(sub, list):
        # update a new dict with substition pairs.
        # if pair is Matrix or MatrixSymbol (N x 1),
        # update each element.
        for old, new in sub:
            if hasattr(old, "__len__") and hasattr(new, "__len__"):
                for elem_old, elem_new in zip(old, new): #type:ignore
                    ret[elem_old] = elem_new
            else:
                try:
                    ret[old] = new
                except Exception as e:
                    raise Exception(e, "\nunhandled case. old and new need to both have '__len__'.")
    else:
        ret = sub
    return ret


def process_var_definition(sub: tuple|list|Matrix) -> dict:
    """This method generates a 'subs' dict from provided
    symbolic variables (Symbol, Function, Matrix). This is
    used for substition of variables into a sympy expression."""
    assert hasattr(sub, "__len__")
    ret = {}
    # for each element get the name
    for var in sub:
        if hasattr(var, "__len__"): # if Matrix
            for elem in var: #type:ignore
                ret[elem] = StateRegister._process_variables(elem)
        else:
            ret[var] = StateRegister._process_variables(var)
    return ret


# def set_matrix_from_var(mat: Matrix, var: Symbol|Function, val: Expr) -> None:
#     id = [i for i in mat if i == var]


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
gravity_x, gravity_y, gravity_z = symbols("gravity_x gravity_y gravity_z")

pos = Matrix([pos_x(t), pos_y(t), pos_z(t)])
vel = Matrix([vel_x(t), vel_y(t), vel_z(t)])
angvel = Matrix([w_x(t), w_y(t), w_z(t)])
q = Matrix([q_0(t), q_1(t), q_2(t), q_3(t)])
mass = symbols("mass")
gravity = Matrix([gravity_x, gravity_y, gravity_z])
speed = symbols("speed")

##################################################
# Inputs
##################################################

acc = Matrix(symbols("acc_x acc_y acc_z"))
torque = Matrix(symbols("torque_x torque_y torque_z"))

wx, wy, wz = angvel
Sw = Matrix([
    [ 0,   wx,  wy,  wz], #type:ignore
    [-wx,  0,  -wz,  wy], #type:ignore
    [-wy,  wz,   0, -wx], #type:ignore
    [-wz, -wy,  wx,   0], #type:ignore
    ])

pos_new = pos + vel * dt
vel_new = vel + (acc + gravity) * dt
angvel_new = angvel + torque * dt
q_new = q + (-0.5 * Sw * q) * dt
mass_new = mass
gravity_new = gravity

##################################################
# Equations for Aerotable / Atmosphere
##################################################

atmosphere = AtmosphereSymbolic()
pos_z_new = pos_new[2]
# gz_dot = -atmosphere.grav_accel(pos_z(t)) #type:ignore
# dynamics[16] = -atmosphere.grav_accel(Symbol("pos_z")) #type:ignore

##################################################
# Subs for differential definitions
##################################################

# gravity finite diff
gacc = atmosphere.grav_accel(pos_z(t)) #type:ignore
gacc_next = atmosphere.grav_accel(pos_z_new) #type:ignore
gacc_dot = atmosphere.grav_accel(pos_z(t)).diff(t) #type:ignore
gacc_delta = gacc_next - gacc #type:ignore
gacc_dot = gacc_delta / dt

# TODO: can we walk designing process by adhearing to
# the chain rule?:
#   - do definitinos satisfy the requirements for
#       the chain rule?
diff_definition = (
        (pos.diff(t),       pos_new.diff(dt)),
        (vel.diff(t),       vel_new.diff(dt)),
        (angvel.diff(t),    angvel_new.diff(dt)),
        (q.diff(t),         q_new.diff(dt)),
        )

# speed
speed_new = sqrt(vel.dot(vel))

##################################################
# Define State Update
##################################################

X_new = Matrix([
    pos,
    vel,
    angvel,
    q,
    mass_new,
    gravity_new,
    speed_new,
    ])

state = Matrix([pos, vel, angvel, q, mass, gravity, speed])
input = Matrix([acc, torque])


##################################################
# Process differential & state definitions
# to substition format
##################################################

diff_sub = process_subs(diff_definition)
state_sub = process_var_definition(state)
input_sub = process_var_definition(input)

##################################################
# Define dynamics
##################################################

dynamics = Matrix(X_new.diff(t))
# dynamic expression found via finite difference
# added here:
# dynamics[16] = gacc_dot

# do substitions
dynamics = dynamics.subs(diff_sub).doit()
dynamics = dynamics.subs(state_sub).subs(input_sub)

model = MissileGeneric().from_expression(dt, state, input, dynamics,
                                         modules=atmosphere.modules)


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

