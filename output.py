import os
import matplotlib.pyplot as plt
from util import norm
from util import unitize
import numpy as np


class OutputManager:
    dir = "./data"
    

    def __init__(self, args, config, t, y, points: list=[]) -> None:
        self.args = args
        self.config = config
        self.t = t
        self.y = y
        self.points = points
        self.velmag = [norm(i) for i in y[:, 3:6]]
        yy = np.array([0, -1, 0])
        self.theta = [np.degrees(np.arccos(np.dot(yy, unitize(vm)))) for vm in y[:, 3:6]]


    def plots(self):
        if self.args.plot_3d:
            # 3D Plot
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d')
            ax.plot3D(self.y[:, 0], self.y[:, 1], self.y[:, 2])
            # ax.plot3D(*targ_R0, marker='.')
            # ax.plot3D(*r_pop1, marker='.', color='green')
            # ax.plot3D(*r_pop2, marker='.', color='red')
            for point in self.points:
                ax.plot3D(*point, marker='.')
            ax.set_xlabel("E")
            ax.set_ylabel("N")
            ax.set_zlabel("D")
            if self.args.save:
                fig.savefig(os.path.join(self.dir, "3d.png"))

        if self.args.plot:
            num_plots = len(self.config)
            fig, axs = plt.subplots(num_plots, figsize=(10, 8), squeeze=True)
            fig.tight_layout()
            plt.subplots_adjust(left=0.08, bottom=0.07, hspace=0.4)

            for iax, nplot in enumerate(self.config):
                title = nplot
                x_axis = self.config[nplot].get("x_axis", "")
                y_axis = self.config[nplot].get("y_axis", "")
                # Choice of X-axis plot
                match x_axis.lower():
                    case 'time' :
                        X = self.t
                        xlabel = 't (s)'
                    case 'alt' :
                        X = self.y[:, 2]
                        xlabel = 'Alt (m)'
                    case 'north' :
                        X = self.y[:, 1]
                        xlabel = 'N (m)'
                    case 'east' :
                        X = self.y[:, 0]
                        xlabel = 'E (m)'
                    case 'alt_dot' :
                        X = self.y[:, 5]
                        xlabel = 'Alt (m/s)'
                    case 'north_dot' :
                        X = self.y[:, 4]
                        xlabel = 'N (m/s)'
                    case 'east_dot' :
                        X = self.y[:, 3]
                        xlabel = 'E (m/s)'
                    case _ :
                        X = self.y[:, 1]
                        xlabel = 'N (m)'

                # Choice of X-axis plot
                match y_axis.lower():
                    case 'alt' :
                        Y = self.y[:, 2]
                        ylabel = 'Alt (m)'
                    case 'north' :
                        Y = self.y[:, 1]
                        ylabel = 'N (m)'
                    case 'east' :
                        Y = self.y[:, 0]
                        ylabel = 'E (m)'
                    case 'alt_dot' :
                        y = self.y[:, 5]
                        ylabel = 'Alt (m/s)'
                    case 'north_dot' :
                        Y = self.y[:, 4]
                        ylabel = 'N (m/s)'
                    case 'east_dot' :
                        Y = self.y[:, 3]
                        ylabel = 'E (m/s)'
                    case 'speed' :
                        Y = self.velmag
                        ylabel = 'Speed (m/s)'
                    case _ :
                        Y = self.y[:, 1]
                        ylabel = 'N (m)'

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
