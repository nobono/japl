import os
import matplotlib.pyplot as plt


class OutputManager:
    dir = "./data"
    

    def __init__(self, args, t, y, points: list=[]) -> None:
        self.args = args
        self.t = t
        self.y = y
        self.points = points
        # velmag = [scipy.linalg.norm(i) for i in y[:, 2:4]]


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
            fig2, (ax2, ax3, ax4) = plt.subplots(3, figsize=(10, 8))
            ax2.plot(self.y[:, 1], self.y[:, 2])
            ax2.set_title("z")
            ax3.plot(self.y[:, 1], self.y[:, 4])
            ax3.set_title("yvel")
            ax4.plot(self.y[:, 1], self.y[:, 5])
            ax4.set_title("zvel")
            if self.args.save:
                fig2.savefig(os.path.join(self.dir, "p.png"))

        plt.show()
