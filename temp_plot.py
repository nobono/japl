import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
# from pyqtgraph import PlotWidget
import numpy as np
from japl import PyQtGraphPlotter

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from pyqtgraph import QtGui
from pyqtgraph import QtWidgets
from pyqtgraph import PlotWidget

import matplotlib.colors as colors
import matplotlib.pyplot as plt


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

    plotter = PyQtGraphPlotter(Nt=10, interval_ms=10, figsize=(10,8))
    plotter.setup([])

    # plotter.set_lim([-2, 2, -2, 2])
    # QtWidgets.QShortcut
    color = colors.to_rgb(colors.TABLEAU_COLORS['tab:orange'])
    color = (color[0]*255, color[1]*255, color[2]*255)

    x = np.linspace(1, 100, 1000)
    y = np.sin(x)

    # line = pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(color, width=2), symbol=None)
    # scatter = pg.ScatterPlotItem(x=x, y=np.cos(x), pen=pg.mkPen(color, width=1), symbol='o')
    # plotter.widget.addItem(line)
    # plotter.widget.addItem(scatter)
    plotter.plot(x, y, color="tab:blue", linewidth=3)

    plotter.show()

