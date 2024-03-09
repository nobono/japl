import os
import matplotlib.pyplot as plt
from util import norm
from util import unitize
import numpy as np


class OutputManager:
    dir = "./data"
    

    def __init__(self, args, t, y, points: list=[]) -> None:
        self.args = args
        self.t = t
        self.y = y
        self.points = points
        self.velmag = [norm(i) for i in y[:, 3:6]]
        yy = np.array([0, -1, 0])
        self.theta = [np.degrees(np.arccos(np.dot(yy, unitize(vm)))) for vm in y[:, 3:6]]


    def plots(self, x_axis: str=""):
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
            fig, axs = plt.subplots(3, figsize=(12, 10), squeeze=True)
            fig.tight_layout()
            plt.subplots_adjust(left=0.06, bottom=0.07, hspace=0.4)

            # Choice of X-axis plot
            match x_axis:
                case 't' :
                    X = self.t
                    xlabel = 't (s)'
                case 'N' :
                    X = self.y[:, 1]
                    xlabel = 'N (m)'
                case _ :
                    X = self.y[:, 1]
                    xlabel = 'N (m)'

            axs[0].set_title("yz")
            axs[0].plot(X, self.y[:, 2])
            axs[0].set_xlabel(xlabel)

            axs[1].set_title("xy")
            axs[1].plot(X, self.y[:, 0])
            axs[1].set_xlabel(xlabel)

            axs[2].set_title("velmag")
            axs[2].plot(X, self.velmag)
            axs[2].set_xlabel(xlabel)

            # axs[3].set_title("theta")
            # axs[3].plot(X, self.theta)

            if self.args.save:
                fig.savefig(os.path.join(self.dir, "p.png"))

        plt.show()
