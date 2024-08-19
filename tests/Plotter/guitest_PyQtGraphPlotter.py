from japl.Plotter.PyQtGraphPlotter import PyQtGraphPlotter
from japl import SimObject
from japl import Sim
import pyqtgraph as pg
import time



class TestPyQtGraphPlotter():


    # def __init__(self) -> None:
    #     self.plotter = PyQtGraphPlotter()


    def test_setup(self):
        plotter = PyQtGraphPlotter()
        assert plotter.app


    def test_create_window(self):
        plotter = PyQtGraphPlotter()
        win = plotter.create_window()
        win = plotter.create_window()
        assert (len(plotter.wins) == 2)
        assert (len(plotter.shortcuts) == 2)
        assert isinstance(win, pg.GraphicsLayoutWidget)


    def test_add_plot_to_window(self):
        plotter = PyQtGraphPlotter()
        win = plotter.create_window()
        color = "blue"
        plot_item = plotter.add_plot_to_window(win,
                                               title="test",
                                               row=0,
                                               col=0,
                                               color=color,
                                               size=1,
                                               aspect="equal")
        assert isinstance(plot_item, pg.PlotDataItem)


    def test_plot(self):
        # adding multiple plot, plot on the
        # same plotItem.
        plotter = PyQtGraphPlotter()
        x = [0, 1, 2, 3]
        y = [0, 1, 2, 3]
        plotter.plot(x, y, color="blue")
        plotter.scatter(x, y, color="red")
        assert len(plotter.wins[0].ci.items) == 1


    def test_FuncAnimation(self):
        def func(frame):
            return frame
        plotter = PyQtGraphPlotter()
        assert not plotter.timer
        plotter.FuncAnimation(func=func, frames=10, interval=100)
        assert plotter.timer


    def test_animate_sim(self):
        sim = Sim([0, 1], 0.1, [])
        plotter = PyQtGraphPlotter()
        plotter.animate(sim)
        assert (plotter.Nt == sim.Nt)
        assert (plotter.dt == sim.dt)
        assert (plotter.simobjs == sim.simobjs)


if __name__ == "__main__":
    test = TestPyQtGraphPlotter()
    test_case_names = [i for i in dir(test) if "test_" in i]
    test_cases = [getattr(test, i) for i in test_case_names]
    print("Testing PyQtGraphPlotter:\n")
    num = 0
    start = time.time()
    for i, case in enumerate(test_cases):
        case()
        print(f"PASS: {test_case_names[i]}")
        num = i
    exec_time = time.time() - start
    print()
    print('=' * 30, end="")
    print(f" {num + 1} Passed in {round(exec_time, 2)}s ", end="")
    print('=' * 30)
