from sympy import Matrix, Symbol, symbols, sqrt, cse
import numpy as np
from code_gen import *



def quat2Rot(q):
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    Rot = Matrix([[1 - 2*(q2**2 + q3**2), 2*(q1*q2 - q0*q3)    , 2*(q1*q3 + q0*q2)    ],
                 [2*(q1*q2 + q0*q3)     , 1 - 2*(q1**2 + q3**2), 2*(q2*q3 - q0*q1)    ],
                 [2*(q1*q3-q0*q2)       , 2*(q2*q3 + q0*q1)    , 1 - 2*(q1**2 + q2**2)]])
    return Rot


def quat_mult(p,q):
    r = Matrix([p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3],
                p[0] * q[1] + p[1] * q[0] + p[2] * q[3] - p[3] * q[2],
                p[0] * q[2] - p[1] * q[3] + p[2] * q[0] + p[3] * q[1],
                p[0] * q[3] + p[1] * q[2] - p[2] * q[1] + p[3] * q[0]])
    return r


def create_cov_matrix(i, j):
    if j >= i:
        # return Symbol("P(" + str(i) + "," + str(j) + ")", real=True)
        # legacy array format
        return Symbol("P[" + str(i) + "][" + str(j) + "]", real=True)
    else:
        return 0


def create_symmetric_cov_matrix(n):
    # define a symbolic covariance matrix
    P = Matrix(n,n,create_cov_matrix)
    for index in range(n):
        for j in range(n):
            if index > j:
                P[index,j] = P[j,index]
    return P


def generate_kalman_gain_equations(P, state, observation, variance, varname = "K"):
    H = Matrix([observation]).jacobian(state)
    innov_var = H * P * H.T +  Matrix([variance])
    K = (P * H.T) / innov_var[0, 0]
    K_simple = cse(K, symbols(f"{varname}0:1000"), optimizations="basic")
    return K_simple


dt = symbols("dt", real=True)
g = symbols("g", real=True)


r_hor_vel = symbols("R_hor_vel", real=True) # horizontal velocity noise variance
r_ver_vel = symbols("R_vert_vel", real=True) # vertical velocity noise variance
r_hor_pos = symbols("R_hor_pos", real=True) # horizontal position noise variance

# inputs, integrated gyro measurements
# delta angle x y z
d_ang_x, d_ang_y, d_ang_z = symbols("dax day daz", real=True)  # delta angle x
d_ang = Matrix([d_ang_x, d_ang_y, d_ang_z])

# inputs, integrated accelerometer measurements
# delta velocity x y z
d_v_x, d_v_y, d_v_z = symbols("dvx dvy dvz", real=True)
d_v = Matrix([d_v_x, d_v_y,d_v_z])

u = Matrix([d_ang, d_v])

# input noise
d_ang_x_var, d_ang_y_var, d_ang_z_var = symbols("daxVar dayVar dazVar", real=True)

d_v_x_var, d_v_y_var, d_v_z_var = symbols("dvxVar dvyVar dvzVar", real=True)

var_u = Matrix.diag(d_ang_x_var, d_ang_y_var, d_ang_z_var, d_v_x_var, d_v_y_var, d_v_z_var)

# define state vector

# attitude quaternion
q0, q1, q2, q3 = symbols("q0 q1 q2 q3", real=True)
q = Matrix([q0,q1,q2,q3])
R_to_earth = quat2Rot(q)
R_to_body = R_to_earth.T

# velocity in NED local frame (north, east, down)
vx, vy, vz = symbols("vn ve vd", real=True)
v = Matrix([vx,vy,vz])

# position in NED local frame (north, east, down)
px, py, pz = symbols("pn pe pd", real=True)
p = Matrix([px,py,pz])

# delta angle bias x y z
d_ang_bx, d_ang_by, d_ang_bz = symbols("dax_b day_b daz_b", real=True)
d_ang_b = Matrix([d_ang_bx, d_ang_by, d_ang_bz])
d_ang_true = d_ang - d_ang_b

# delta velocity bias x y z
d_vel_bx, d_vel_by, d_vel_bz = symbols("dvx_b dvy_b dvz_b", real=True)
d_vel_b = Matrix([d_vel_bx, d_vel_by, d_vel_bz])
d_vel_true = d_v - d_vel_b

# state vector at arbitrary time t
state = Matrix([q, v, p, d_ang_b, d_vel_b])

print('Defining state propagation ...')
# kinematic processes driven by IMU 'control inputs'
q_new = quat_mult(q, Matrix([1, 0.5 * d_ang_true[0],  0.5 * d_ang_true[1],  0.5 * d_ang_true[2]]))
v_new = v + R_to_earth * d_vel_true + Matrix([0,0,g]) * dt
p_new = p + v * dt

# static processes
d_ang_b_new = d_ang_b
d_vel_b_new = d_vel_b

# predicted state vector at time t + dt
state_new = Matrix([q_new, v_new, p_new, d_ang_b_new, d_vel_b_new])

print('Computing state propagation jacobian ...')
A = state_new.jacobian(state)
G = state_new.jacobian(u)

P = create_symmetric_cov_matrix(len(state))

print('Computing covariance propagation ...')
P_new = A * P * A.T + G * var_u * G.T

for index in range(len(state)):
    for j in range(len(state)):
        if index > j:
            P_new[index,j] = 0

print('Simplifying covariance propagation ...')
P_new_simple = cse(P_new, symbols("PS0:400"), optimizations='basic')

args = symbols("q0, q1, q2, q3,"            # quaternion
               "vn, ve, vd,"                # velocity in NED local frame
               "pn, pe, pd,"                # position in NED local frame
               "dvx, dvy, dvz,"             # delta velocity (accelerometer measurements)
               "dax, day, daz,"             # delta angle (gyroscope measurements)
               "dax_b, day_b, daz_b,"       # delta angle bias
               "dvx_b, dvy_b, dvz_b,"       # delta velocity bias
               "P,"                         # covariance matrix
               "daxVar, dayVar, dazVar,"    # gyro input noise
               "dvxVar, dvyVar, dvzVar,"    # accel input noise
               "dt")

print('Writing covariance propagation to file ...')
cov_code_generator = OctaveCodeGenerator("./generated/cov_predict.m")
cov_code_generator.print_string("Equations for covariance matrix prediction, without process noise!")
cov_code_generator.write_function_definition(name="cov_predict",
                                             args=args,
                                             returns=["nextP"])
cov_code_generator.write_subexpressions(P_new_simple[0])
cov_code_generator.write_matrix(matrix=Matrix(P_new_simple[1]),
                                variable_name="nextP",
                                is_symmetric=True,
                                pre_bracket="(",
                                post_bracket=")",
                                separator=", ")

cov_code_generator.close()

