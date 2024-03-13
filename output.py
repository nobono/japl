import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from util import norm
from util import unitize
from scipy import constants



class OutputManager:
    dir = "./data"
    

    def __init__(self, args, config, t, y, points: list=[]) -> None:
        self.args = args
        self.config = config
        self.t = t
        self.y = y
        self.points = points
        self.velmag = np.asarray([norm(i) for i in y[:, 3:6]])
        self.accmag = np.asarray([norm(i) for i in y[:, 6:9]])
        self.G = np.asarray([i / constants.g for i in self.accmag])
        yy = np.array([0, -1, 0])
        self.theta = [np.degrees(np.arccos(np.dot(yy, unitize(vm)))) for vm in y[:, 3:6]]


    def axis_selection(self, name: str) -> tuple[np.ndarray, str]:
        match name.lower():
            case 'time' :
                Y = self.t
                label = 'Time (s)'
            case 'alt' :
                Y = self.y[:, 2]
                label = 'Alt (m)'
            case 'north' :
                Y = self.y[:, 1]
                label = 'N (m)'
            case 'east' :
                Y = self.y[:, 0]
                label = 'E (m)'
            case 'alt_dot' :
                Y = self.y[:, 5]
                label = 'Alt vel (m/s)'
            case 'north_dot' :
                Y = self.y[:, 4]
                label = 'N vel (m/s)'
            case 'east_dot' :
                Y = self.y[:, 3]
                label = 'E vel (m/s)'
            case 'speed' :
                Y = self.velmag
                label = 'Speed (m/s)'
            case 'alt_dot_dot' :
                Y = self.y[:, 8]
                label = 'Alt acc (m/s^2)'
            case 'north_dot_dot' :
                Y = self.y[:, 7]
                label = 'North acc (m/s^2)'
            case 'east_dot_dot' :
                Y = self.y[:, 6]
                label = 'East acc (m/s^2)'
            case 'accel' :
                Y = self.accmag
                label = 'accel mag (m/s^2)'
            case 'g' :
                Y = self.G
                label = 'Gs'
            case _ :
                Y = self.y[:, 1]
                label = 'N (m)'

        return Y, label



    def plots(self):
        if self.args.plot_3d:
            # 3D Plot
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d', aspect='equal')
            ax.plot3D(self.y[:, 0], self.y[:, 1], self.y[:, 2])
            ax.set_xlabel("E")
            ax.set_ylabel("N")
            ax.set_zlabel("D")

            # Setup sliders
            ax_zlim = fig.add_axes([0.25, 0.0, 0.65, 0.03]) #type:ignore
            slider_zlim = Slider(
                    ax=ax_zlim,
                    label="zlim",
                    valmin = 5.0,
                    valmax=30e3,
                    valinit=1.0,
                    )

            def update(val):
                zlim = ax.get_zlim()
                ax.set_zlim([zlim[0], val])
                fig.canvas.draw_idle()

            slider_zlim.on_changed(update)

            # scale x-axis same as y-axis
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ylim_delta = abs(ylim[0] - ylim[1])
            ax.set_xlim(xlim[0] - ylim_delta/2, xlim[1] + ylim_delta/2)

            # Plot user-specified points
            for point in self.points:
                ax.plot3D(*point, marker='.')

            # Save to output file
            if self.args.save:
                fig.savefig(os.path.join(self.dir, "3d.png"))

        if self.args.plot:
            num_plots = len(self.config)
            fig, axs = plt.subplots(num_plots, figsize=(10, 8), squeeze=True)
            fig.tight_layout()
            plt.subplots_adjust(left=0.08, bottom=0.07, hspace=0.4)

            for iax, nplot in enumerate(self.config):
                title = nplot
                x_axis_selection = self.config[nplot].get("x_axis", "")
                y_axis_selection = self.config[nplot].get("y_axis", "")

                # Choice of XY-axis plots
                X, xlabel = self.axis_selection(x_axis_selection)
                Y, ylabel = self.axis_selection(y_axis_selection)

                if num_plots > 1:
                    axs[iax].set_title(title)
                    axs[iax].plot(X, Y)
                    axs[iax].set_xlabel(xlabel)
                    axs[iax].set_ylabel(ylabel)
                else:
                    axs.set_title(title)
                    axs.plot(X, Y)
                    axs.set_xlabel(xlabel)
                    axs.set_ylabel(ylabel)


                # axs[1].set_title("xy")
                # axs[1].plot(X, self.y[:, 0])
                # axs[1].set_xlabel(xlabel)

                # axs[2].set_title("velmag")
                # axs[2].plot(X, self.velmag)
                # axs[2].set_xlabel(xlabel)

                # axs[3].set_title("theta")
                # axs[3].plot(X, self.theta)

            if self.args.save:
                fig.savefig(os.path.join(self.dir, "p.png"))

        plt.show()
