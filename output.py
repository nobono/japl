import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import Animation
from matplotlib.widgets import Button

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
    

    # def animate(self):



    def plots(self):
        if self.args.plot_3d:
            # 3D Plot
            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection='3d', aspect='equal')

            bcamera_tracking = False

            # animation objs
            vel_vec = np.dstack([self.y[0, :3], self.y[0, :3] + self.y[0, 3:6]]).squeeze()
            acc_vec = np.dstack([self.y[0, :3], self.y[0, :3] + self.y[0, 6:9]]).squeeze()
            vel_vec_marker = ax.plot3D(*vel_vec, color="orange")
            acc_vec_marker = ax.plot3D(*acc_vec, color="green")
            pos_marker = ax.plot3D(*self.y[0, :3], marker='.', color="red", markersize=5)

            ax.plot3D(self.y[:, 0], self.y[:, 1], self.y[:, 2])
            ax.set_xlabel("E")
            ax.set_ylabel("N")
            ax.set_zlabel("U")

            # Buttons
            ax_tracking = fig.add_axes([0.01, 0.03, 0.15, 0.075])
            button_camera_track_enable = Button(ax_tracking, "Camera Tracking")

            def enable_tracking_on_click(val):
                nonlocal bcamera_tracking
                bcamera_tracking = not bcamera_tracking

            button_camera_track_enable.on_clicked(enable_tracking_on_click)

            # Setup sliders
            ax_zlim = fig.add_axes([0.25, 0.0, 0.65, 0.03]) #type:ignore
            ax_time = fig.add_axes([0.25, 0.03, 0.65, 0.03]) #type:ignore

            slider_zlim = Slider(
                    ax=ax_zlim,
                    label="zlim",
                    valmin=5.0,
                    valmax=20e3,
                    valinit=1.0,
                    )
            slider_time = Slider(
                    ax=ax_time,
                    label="time",
                    valmin=0,
                    valmax=len(self.t),
                    valinit=0,
                    )

            def update_zlim(val):
                zlim = ax.get_zlim()
                ax.set_zlim([0, val])
                fig.canvas.draw_idle()

            def update_pos(val):
                pos_marker[0]._verts3d = self.y[val, :3]

            def update_vel_vec(val, scale):
                pos = self.y[val, :3]
                vel = self.y[val, 3:6]
                vel_vec = np.dstack([pos, (pos + vel * scale)]).squeeze()
                vel_vec_marker[0]._verts3d = vel_vec

            def update_acc_vec(val, scale):
                pos = self.y[val, :3]
                acc = self.y[val, 6:9]
                acc_vec = np.dstack([pos, (pos + acc * scale)]).squeeze()
                acc_vec_marker[0]._verts3d = acc_vec

            def update_time(val):
                val = int(val)
                unit_division = 100
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                zlim = ax.get_zlim()
                xlim_len = np.fabs(xlim[1] - xlim[0])
                ylim_len = np.fabs(ylim[1] - ylim[0])
                zlim_len = np.fabs(zlim[1] - zlim[0])
                xlim_scale = xlim_len / unit_division
                ylim_scale = ylim_len / unit_division
                zlim_scale = zlim_len / unit_division
                scale = np.linalg.norm(np.array([xlim_scale, ylim_scale])) / 10
                ###################################
                # if camera tracking position
                ###################################
                if bcamera_tracking:
                    pos = self.y[val, :3]
                    view = ax._get_view()
                    xscale = xlim_len / 2
                    yscale = ylim_len / 2
                    zscale = zlim_len / 2
                    xview = (pos[0] - xscale, pos[0] + xscale)
                    yview = (pos[1] - yscale, pos[1] + yscale)
                    zview = (pos[2] - zscale, pos[2] + zscale)
                    view[0]['xlim'] = xview
                    view[0]['ylim'] = yview
                    view[0]['zlim'] = zview
                    ax._set_view(view)
                ###################################
                update_pos(val)
                update_vel_vec(val, scale)
                update_acc_vec(val, scale)
                fig.canvas.draw_idle()

            slider_zlim.on_changed(update_zlim)
            slider_time.on_changed(update_time)

            # scale x-axis same as y-axis
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            zlim = ax.get_zlim()
            ylim_delta = abs(ylim[0] - ylim[1])
            ax.set_xlim(xlim[0] - ylim_delta/2, xlim[1] + ylim_delta/2)
            ax.set_zlim(zlim[0] - ylim_delta/2, zlim[1] + ylim_delta/2)

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

            if self.args.save:
                fig.savefig(os.path.join(self.dir, "p.png"))

        plt.show()
