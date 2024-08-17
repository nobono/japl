# import unittest
# from japl import PyQtGraphPlotter
# from japl import SimObject
# import pyqtgraph as pg



# class TestPyQtGraphPlotter_1(unittest.TestCase):


#     def setUp(self) -> None:
#         self.plotter = PyQtGraphPlotter()


#     def test_instantiate_add_window(self):
#         # plotter = PyQtGraphPlotter()
#         win = self.plotter.add_window()
#         self.assertEqual(len(self.plotter.wins), 1)
#         self.assertEqual(len(self.plotter.shortcuts), 1)
#         self.assertIsInstance(win, pg.GraphicsLayoutWidget)
#         # plotter.app.exit()


#     def test_instantiate_add_plot(self):
#         # plotter = PyQtGraphPlotter()
#         win = self.plotter.add_window()
#         color_code = next(self.plotter.color_cycle)
#         plot_item = self.plotter.add_plot(win,
#                                      title="test",
#                                      row=0,
#                                      col=0,
#                                      color_code=color_code,
#                                      size=1,
#                                      aspect="equal")
#         self.assertIsInstance(plot_item, pg.PlotDataItem)
#         # plotter.app.exit()


# if __name__ == '__main__':
#     unittest.main()
