import numpy as np
from sympy import Symbol, Matrix, symbols
from sympy import MatrixSymbol
from japl import PyQtGraphPlotter



# dt = Symbol('dt', real=True)
# pos_n = Symbol("pos_n", real=True)  # (t)
# pos_e = Symbol("pos_e", real=True)  # (t)
# pos_d = Symbol("pos_d", real=True)  # (t)
# vel_n = Symbol("vel_n", real=True)  # (t)
# vel_e = Symbol("vel_e", real=True)  # (t)
# vel_d = Symbol("vel_d", real=True)  # (t)
# acc_n = Symbol("acc_n", real=True)  # (t)
# acc_e = Symbol("acc_e", real=True)  # (t)
# acc_d = Symbol("acc_d", real=True)  # (t)
# pos = Matrix([pos_n, pos_e, pos_d])
# vel = Matrix([vel_n, vel_e, vel_d])
# acc = Matrix([acc_n, acc_e, acc_d])

# z_pos_x, z_pos_y, z_pos_z = symbols("z_pos_x, z_pos_y, z_pos_z", real=True)
# z_vel_x, z_vel_y, z_vel_z = symbols("z_vel_x, z_vel_y, z_vel_z", real=True)
# z_pos = Matrix([z_pos_x, z_pos_y, z_pos_z])
# z_vel = Matrix([z_vel_x, z_vel_y, z_vel_z])

# pos_x_var, pos_y_var, pos_z_var = symbols('pos_x_var, pos_y_var, pos_z_var')
# vel_x_var, vel_y_var, vel_z_var = symbols('vel_x_var, vel_y_var, vel_z_var')
# input_var = Matrix.diag([pos_x_var, pos_y_var, pos_z_var,
#                          vel_x_var, vel_y_var, vel_z_var])

# pos_new = pos + vel * dt  # + 0.5 * acc_world_measured * dt**2
# vel_new = vel  # + acc_world_measured * dt

# state = Matrix([pos, vel])
# state_new = Matrix([pos_new, vel_new])
# input = Matrix([z_pos, z_vel])

# F = state_new.jacobian(state)
# G = state_new.jacobian(input)
# X_new = F * state + G * input
# Q = G * input_var * G.T

# P = MatrixSymbol("P", len(state), len(state)).as_mutable()
# # make P symmetric
# for index in range(P.shape[0]):
#     for j in range(P.shape[0]):
#         if index > j:
#             P[index, j] = P[j, index]

# P_new = F * P * F.T + Q

# # alpha = P_new / (P_new + R)
# # beta ~= alpha * dt

# acc_n_var, acc_e_var, acc_d_var = symbols("acc_n_var, acc_e_var, acc_d_var")
# acc_var = Matrix([acc_n_var, acc_e_var, acc_d_var])
# W = Matrix([0.5 * acc * dt**2, acc * dt])
# Q = (W * W.T).subs({acc_n**2: acc_n_var})\
#         .subs({acc_e**2: acc_e_var})\
#         .subs({acc_d**2: acc_d_var})\


# def calc_process_noise(acc_var: np.ndarray, dt: float):
#     acc_n, acc_e, acc_d = acc
#     acc_n_var, acc_e_var, acc_d_var = acc_var
#     Q = np.array([[0.25*acc_n_var*dt**4,   0.25*acc_e*acc_n*dt**4, 0.25*acc_d*acc_n*dt**4, 0.5*acc_n_var*dt**3,   0.5*acc_e*acc_n*dt**3, 0.5*acc_d*acc_n*dt**3],  # type:ignore # noqa
#                   [0.25*acc_e*acc_n*dt**4, 0.25*acc_e_var*dt**4,   0.25*acc_d*acc_e*dt**4, 0.5*acc_e*acc_n*dt**3, 0.5*acc_e_var*dt**3,   0.5*acc_d*acc_e*dt**3],  # type:ignore # noqa
#                   [0.25*acc_d*acc_n*dt**4, 0.25*acc_d*acc_e*dt**4, 0.25*acc_d_var*dt**4,   0.5*acc_d*acc_n*dt**3, 0.5*acc_d*acc_e*dt**3, 0.5*acc_d_var*dt**3],  # type:ignore # noqa
#                   [0.5*acc_n_var*dt**3,    0.5*acc_e*acc_n*dt**3,  0.5*acc_d*acc_n*dt**3,  acc_n_var*dt**2,       acc_e*acc_n*dt**2,     acc_d*acc_n*dt**2],  # type:ignore # noqa
#                   [0.5*acc_e*acc_n*dt**3,  0.5*acc_e_var*dt**3,    0.5*acc_d*acc_e*dt**3,  acc_e*acc_n*dt**2,     acc_e_var*dt**2,       acc_d*acc_e*dt**2],  # type:ignore # noqa
#                   [0.5*acc_d*acc_n*dt**3,  0.5*acc_d*acc_e*dt**3,  0.5*acc_d_var*dt**3,    acc_d*acc_n*dt**2,     acc_d*acc_e*dt**2,     acc_d_var*dt**2]])  # type:ignore # noqa
#     return Q



def alpha_beta_filter(pos: np.ndarray, vel: np.ndarray, z_pos: np.ndarray,
                      variance: float, noise: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    R = noise
    Q = variance
    alpha = Q / (Q + R)
    beta = alpha * dt
    res = z_pos - pos
    pos_new = pos + alpha * res
    vel_new = vel + (beta / dt) * res
    return pos_new, vel_new


if __name__ == "__main__":
    dt = 0.1
    pos0 = np.array([0, 0, 0])
    vel0 = np.array([0, 0, 0])
    pos = pos0
    vel = vel0

    variance = 0.05
    noise = 0.05

    X = []
    Y = []
    Yf = []
    pos = pos0
    vel = vel0
    for i in range(100):
        z_pos = np.random.normal([0, 0, 0], .1)
        pos_new, vel_new = alpha_beta_filter(pos=pos, vel=vel, z_pos=z_pos,
                                             variance=variance, noise=noise, dt=dt)
        X += [i]
        Y += [z_pos[0]]
        Yf += [pos_new[0]]
        pos = pos_new
        vel = vel_new


    plotter = PyQtGraphPlotter(aspect="auto")
    plotter.plot(X, Y)
    plotter.plot(X, Yf)
    plotter.show()
