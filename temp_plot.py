import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
# from pyqtgraph import PlotWidget
import numpy as np
from japl import PyQtGraphPlotter

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from pyqtgraph import GraphicsLayoutWidget, PlotCurveItem, QtGui
from pyqtgraph import QtWidgets
from pyqtgraph import PlotWidget

import matplotlib.colors as colors
import matplotlib.pyplot as plt

from japl.Model.Model import Model
from japl.SimObject.SimObject import SimObject


# x = np.random.normal(size=1000)
# y = np.random.normal(size=1000)

# myplot = pg.plot([], [], pen=None, symbol='o')  ## setting pen=None disables line drawing
# myplot.showGrid(True, True, 0.5)


if __name__ == '__main__':
    ## Always start by initializing Qt (only once per application)
    # app = QtWidgets.QApplication([])

    # ## Define a top-level widget to hold everything
    # w = QtWidgets.QWidget()
    # w.setWindowTitle('PyQtGraph example')

    # ## Create some widgets to be placed inside
    # btn = QtWidgets.QPushButton('press me')
    # text = QtWidgets.QLineEdit('enter text')
    # listw = QtWidgets.QListWidget()
    # plot = pg.PlotWidget()

    # ## Create a grid layout to manage the widgets size and position
    # layout = QtWidgets.QGridLayout()
    # w.setLayout(layout)

    # ## Add widgets to the layout in their proper positions
    # layout.addWidget(btn, 0, 0)  # button goes in upper-left
    # layout.addWidget(text, 1, 0)  # text edit goes in middle-left
    # layout.addWidget(listw, 2, 0)  # list widget goes in bottom-left
    # layout.addWidget(plot, 0, 1, 3, 1)  # plot goes on right side, spanning 3 rows
    # ## Display the widget as a new window
    # w.show()

    ## Start the Qt event loop
    # app.exec()  # or app.exec_() for PyQt5 / PySide2

    ######################
    # x = np.linspace(1, 100, 10000)
    # y = np.sin(x)
    # plt.plot(x, y)
    # plt.show()
    # quit()

    
    A = np.array([
        [0, 0, 0, 1, 0, 0,  0],
        [0, 0, 0, 0, 1, 0,  0],
        [0, 0, 0, 0, 0, 1,  0],
        [0, 0, 0, 0, 0, 0,  0],
        [0, 0, 0, 0, 0, 0,  0],
        [0, 0, 0, 0, 0, 0,  0],

        [0, 0, 0, 0, 0, 0,  1],
        ])
    B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],

        [0, 0, 0],
        ])

    model = Model.ss(A, B)
    vehicle = SimObject(model=model, size=0, color='tab:blue')

    vehicle.register_state("x",         0, "x (m)")
    vehicle.register_state("y",         1, "y (m)")
    vehicle.register_state("z",         2, "z (m)")
    vehicle.register_state("vx",        3, "xvel (m/s)")
    vehicle.register_state("vy",        4, "yvel (m/s)")
    vehicle.register_state("vz",        5, "zvel (m/s)")
    vehicle.register_state("fuel_burn", 6, "Fuel Burn ")

    vehicle.plot.set_config({
                "Pos": {
                    "xaxis": "x",
                    "yaxis": "z",
                    },
                "Vel": {
                    "xaxis": "x",
                    "yaxis": "vz",
                    },
                })

    plotter = PyQtGraphPlotter(Nt=10, figsize=(6, 4))
    plotter.setup([vehicle])
    plotter.show()
    quit()
    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    widget = PlotWidget()

    view = GraphicsLayoutWidget()
    view.addPlot(row=0, col=0, title="Data 1")
    view.addPlot(row=0, col=1, title="Data 2")

    win.setCentralWidget(widget)

    view.show()
    # win.show()

    app.exec_()
    # self.layout.addViewBox(row=1, col=0, colspan=2)
    quit()

    plotter = PyQtGraphPlotter(
            Nt=10,
            interval_ms=10,
            figsize=(10,6),
            antialias=False,
            aspect="auto",
            )
    plotter.setup([])

    # x = np.linspace(1, 10, 1000)
    # y = np.sin(x)

    plotter.plot([], [], linewidth=3)
    # plotter.scatter(x, np.cos(x), linewidth=3)

    Nt = 2000
    x = np.zeros((Nt,))
    y = np.zeros((Nt,))
    # x, y = [], []

    count = 1
    i = 0
    def update():
        global x, y
        global count
        global i
        # x += [count]
        # y += [np.sin(count)]
        x[i] = i * .1
        y[i] = np.sin(i * .1)

        pi = plotter.widget.getPlotItem()
        d: PlotCurveItem = pi.dataItems[0]
        d.setData(x=x[:i], y=y[:i])
        
        count += .1
        i += 1

    # timer = QtCore.QTimer()
    # timer.timeout.connect(update)
    # timer.start(10)
    plotter.FuncAnimation(func=update, frames=-1, interval_ms=10)

    


    plotter.show()

    # ret = plt.rcParams["axes.prop_cycle"]
    # ret = plt.get_cmap("tab10")
    # from matplotlib import colors
    # print(colors.TABLEAU_COLORS)

