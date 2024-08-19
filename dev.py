import sys
import numpy as np
from japl import PyQtGraphPlotter
from japl import SimObject
from japl import Sim
from japl import SimObject
from japl import Model
from sympy import symbols
import pyqtgraph as pg


def func(X, U, dt):
    U[0] = 0.1
    p_dot = X[1]
    v_dot = U[0]
    X_dot = np.array([p_dot, v_dot])
    return X_dot


p, v, a, dt = symbols("p v a dt")

model = Model.from_function(dt, [p, v], [a], func)

simobj = SimObject(model)
simobj.plot.set_config({
    "Position": {
        "xaxis": 't',
        "yaxis": 'p',
        "color": 'blue',
        "size": 2,
        },
    })

sim = Sim([0, 10], 0.1, [simobj])
plotter = PyQtGraphPlotter(frame_rate=25, figsize=[8, 4], show_grid=0)
plotter.animate(sim)

# x = [0, 1, 2, 3]
# y = [0, 1, 2, 3]
# plotter.plot(x, y, color="blue")
# plotter.scatter(x, y, color="red")



plotter.show()
# pg.GraphicsWidget

# print(simobj.Y)


# plotter = PyQtGraphPlotter()
# win = plotter.create_window()
# color_code = next(plotter.color_cycle)
# plot_item = plotter.add_plot(win,
#                              title="test",
#                              row=0,
#                              col=0,
#                              color_code=color_code,
#                              size=1,
#                              aspect="equal")


# plotter.show()
