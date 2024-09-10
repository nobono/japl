import os
import numpy as np
from japl.Util.Matlab import MatFile
from japl import PyQtGraphPlotter
from japl.Util.Results import Results
DIR = os.path.dirname(__file__)
np.set_printoptions(suppress=True, precision=3)


plotter = PyQtGraphPlotter(frame_rate=30,
                           figsize=[10, 6],
                           aspect="auto",
                           background_color="white",
                           text_color="black")


fo = MatFile(DIR + "/../../../data/flyout.mat").flyout  # type:ignore
run1 = Results.load(DIR + "/run1.pickle")
run = run1


col = ("Thrust", "thrust")
plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
             title="Thrust vs. Time",
             ylabel="Thrust (N)",
             xlabel="Time (s)")
plotter.plot(getattr(run, "t"), getattr(run, col[1]))


col = ("Ca", "CA")
plotter.figure()
plotter.plot(getattr(fo, "Time"), getattr(fo, col[0]),
             title="CA vs. Time",
             ylabel="CA",
             xlabel="Time")
plotter.plot(getattr(run, "t"), getattr(run, col[1]))


plotter.show()
