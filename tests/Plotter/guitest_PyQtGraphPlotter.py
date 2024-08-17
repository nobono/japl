# import unittest
from japl import PyQtGraphPlotter
# # from japl import SimObject
import pyqtgraph as pg


# class TestPyQtGraphPlotter_1(unittest.TestCase):


#     # def setUp(self) -> None:
#     #     self.plotter = PyQtGraphPlotter()


#     def test_instantiate_setup(self):
#         plotter = PyQtGraphPlotter()
#         self.assertTrue(plotter.app)


#     def test_instantiate_add_window(self):
#         plotter = PyQtGraphPlotter()
#         win = plotter.add_window()
#         self.assertEqual(len(plotter.wins), 1)
#         self.assertEqual(len(plotter.shortcuts), 1)
#         self.assertIsInstance(win, pg.GraphicsLayoutWidget)


#     def test_instantiate_add_plot(self):
#         plotter = PyQtGraphPlotter()
#         win = plotter.add_window()
#         color_code = next(plotter.color_cycle)
#         plot_item = plotter.add_plot(win,
#                                      title="test",
#                                      row=0,
#                                      col=0,
#                                      color_code=color_code,
#                                      size=1,
#                                      aspect="equal")
#         self.assertIsInstance(plot_item, pg.PlotDataItem)


# if __name__ == '__main__':
#     unittest.main()


class TestPyQtGraphPlotter():


    def test_instantiate_setup(self):
        plotter = PyQtGraphPlotter()
        assert plotter.app


    def test_instantiate_add_window(self):
        plotter = PyQtGraphPlotter()
        win = plotter.add_window()
        assert (len(plotter.wins) == 1)
        assert (len(plotter.shortcuts) == 1)
        assert isinstance(win, pg.GraphicsLayoutWidget)


    def test_instantiate_add_plot(self):
        plotter = PyQtGraphPlotter()
        win = plotter.add_window()
        color_code = next(plotter.color_cycle)
        plot_item = plotter.add_plot(win,
                                     title="test",
                                     row=0,
                                     col=0,
                                     color_code=color_code,
                                     size=1,
                                     aspect="equal")
        assert isinstance(plot_item, pg.PlotDataItem)


if __name__ == "__main__":
    test = TestPyQtGraphPlotter()
    test_case_names = [i for i in dir(test) if "test_" in i]
    test_cases = [getattr(test, i) for i in test_case_names]
    print("Testing PyQtGraphPlotter:")
    for i, case in enumerate(test_cases):
        case()
        print(f"PASS: {test_case_names[i]}")
