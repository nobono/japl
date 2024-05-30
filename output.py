import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import Animation
from matplotlib.widgets import Button

from util import norm
from util import unitize
from scipy import constants

# plt.style.use('seaborn-v0_8-dark')
plt.style.use('bmh')



class OutputManager:
    dir = "./data"
    

    def __init__(self, args, config, t, y, points: list=[], figsize: tuple[float, float]=(10, 8)) -> None:
        self.figsize = figsize
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
        self.register = {}


    def register_state(self, name: str, id: int, label: str=""):
        self.register.update({name: {"id": id, "label": label}})


    def axis_selection(self, name: str) -> tuple[np.ndarray, str]:
        # match name.lower():
        #     case 'time' :
        #         Y = self.t
        #         label = 'Time (s)'
        #     case 'alt' :
        #         Y = self.y[:, 2]
        #         label = 'Alt (m)'
        #     case 'north' :
        #         Y = self.y[:, 1]
        #         label = 'N (m)'
        #     case 'east' :
        #         Y = self.y[:, 0]
        #         label = 'E (m)'
        #     case 'alt_dot' :
        #         Y = self.y[:, 5]
        #         label = 'Alt vel (m/s)'
        #     case 'north_dot' :
        #         Y = self.y[:, 4]
        #         label = 'N vel (m/s)'
        #     case 'east_dot' :
        #         Y = self.y[:, 3]
        #         label = 'E vel (m/s)'
        #     case 'speed' :
        #         Y = self.velmag
        #         label = 'Speed (m/s)'
        #     case 'alt_dot_dot' :
        #         Y = self.y[:, 8]
        #         label = 'Alt acc (m/s^2)'
        #     case 'north_dot_dot' :
        #         Y = self.y[:, 7]
        #         label = 'North acc (m/s^2)'
        #     case 'east_dot_dot' :
        #         Y = self.y[:, 6]
        #         label = 'East acc (m/s^2)'
        #     case 'accel' :
        #         Y = self.accmag
        #         label = 'accel mag (m/s^2)'
        #     case 'g' :
        #         Y = self.G
        #         label = 'Gs'
        #     case _ :
        #         Y = self.y[:, 1]
        #         label = 'N (m)'
        if name == "time":
            Y = self.t
            label = "Time (s)"
        else:
            id = self.register[name]["id"]
            Y = self.y[:, id]
            label = self.register[name]["label"]
            if not label:
                label = name

        return Y, label
    

    def on_key_press(self, event):
        if event.key == 'a':
            self.slider_time.set_val(self.slider_time.val - 1)
        elif event.key == 'd':
            self.slider_time.set_val(self.slider_time.val + 1)

    def update_zlim(self, val):
        self.ax_3d.set_zlim([0, val])
        self.fig_3d.canvas.draw_idle()


    def update_pos(self, val):
        self.pos_marker[0]._verts3d = self.y[val, :3]


    def update_vel_vec(self, val, scale):
        pos = self.y[val, :3]
        vel = self.y[val, 3:6]
        vel_vec = np.dstack([pos, (pos + vel * scale)]).squeeze()
        self.vel_vec_marker[0]._verts3d = vel_vec


    def update_acc_vec(self, val, scale):
        pos = self.y[val, :3]
        acc = self.y[val, 6:9]
        acc_vec = np.dstack([pos, (pos + acc * scale)]).squeeze()
        self.acc_vec_marker[0]._verts3d = acc_vec


    def update_camera_tracking(self, pos):
        view = self.ax_3d._get_view()
        xlim = self.ax_3d.get_xlim()
        ylim = self.ax_3d.get_ylim()
        zlim = self.ax_3d.get_zlim()
        xlim_len = np.fabs(xlim[1] - xlim[0])
        ylim_len = np.fabs(ylim[1] - ylim[0])
        zlim_len = np.fabs(zlim[1] - zlim[0])
        xscale = xlim_len / 2
        yscale = ylim_len / 2
        zscale = zlim_len / 2
        xview = (pos[0] - xscale, pos[0] + xscale)
        yview = (pos[1] - yscale, pos[1] + yscale)
        zview = (pos[2] - zscale, pos[2] + zscale)
        if len(view) > 2:
            new_view = (xview, yview, zview, view[3], view[4], view[5])
            self.ax_3d._set_view(new_view)
        else:
            view[0]['xlim'] = xview
            view[0]['ylim'] = yview
            view[0]['zlim'] = zview
            self.ax_3d._set_view(view)


    def update_time(self, val):
        val = int(val)
        unit_division = 100
        xlim = self.ax_3d.get_xlim()
        ylim = self.ax_3d.get_ylim()
        zlim = self.ax_3d.get_zlim()
        xlim_len = np.fabs(xlim[1] - xlim[0])
        ylim_len = np.fabs(ylim[1] - ylim[0])
        zlim_len = np.fabs(zlim[1] - zlim[0])
        xlim_scale = xlim_len / unit_division
        ylim_scale = ylim_len / unit_division
        zlim_scale = zlim_len / unit_division
        scale = np.linalg.norm(np.array([xlim_scale, ylim_scale])) / 10.0 #type:ignore
        # if camera tracking position
        if self.bcamera_tracking:
            pos = self.y[val, :3]
            self.update_camera_tracking(pos)
        # call update fnncs
        self.update_pos(val)
        self.update_vel_vec(val, scale)
        self.update_acc_vec(val, scale)
        self.fig_3d.canvas.draw_idle()


    def enable_tracking_on_click(self, val):
        self.bcamera_tracking = not self.bcamera_tracking
        if self.bcamera_tracking:
            self.button_camera_track_enable.color = "grey"
            self.button_camera_track_enable.label.set_text("Camera Tracking On")
        else:
            self.button_camera_track_enable.color = "white"
            self.button_camera_track_enable.label.set_text("Camera Tracking Off")


    def plots(self):
        if self.args.plot_3d:
            # 3D Plot
            self.fig_3d = plt.figure(figsize=self.figsize)
            self.ax_3d = plt.axes(projection='3d', aspect='equal')

            self.bcamera_tracking = False

            # animation objs
            _vel_vec = np.dstack([self.y[0, :3], self.y[0, :3] + self.y[0, 3:6]]).squeeze()
            _acc_vec = np.dstack([self.y[0, :3], self.y[0, :3] + self.y[0, 6:9]]).squeeze()
            self.vel_vec_marker = self.ax_3d.plot3D(*_vel_vec, color="orange")
            self.acc_vec_marker = self.ax_3d.plot3D(*_acc_vec, color="green")
            self.pos_marker = self.ax_3d.plot3D(*self.y[0, :3], marker='.', color="red", markersize=5)

            self.ax_3d.plot3D(self.y[:, 0], self.y[:, 1], self.y[:, 2])
            self.ax_3d.set_xlabel("E")
            self.ax_3d.set_ylabel("N")
            self.ax_3d.set_zlabel("U")

            # Setup keyboard event callback
            self.fig_3d.canvas.mpl_connect('key_press_event', self.on_key_press)

            # Buttons
            ax_tracking = self.fig_3d.add_axes((0.01, 0.03, 0.15, 0.075))
            self.button_camera_track_enable = Button(ax_tracking, "Camera Tracking Off")

            self.button_camera_track_enable.on_clicked(self.enable_tracking_on_click)

            # Setup sliders
            ax_zlim = self.fig_3d.add_axes((0.25, 0.0, 0.65, 0.03))
            ax_time = self.fig_3d.add_axes((0.25, 0.03, 0.65, 0.03))

            slider_zlim = Slider(
                    ax=ax_zlim,
                    label="zlim",
                    valmin=5.0,
                    valmax=20e3,
                    valinit=self.y[:, 2].max(),
                    )
            self.slider_time = Slider(
                    ax=ax_time,
                    label="time",
                    valmin=0,
                    valmax=len(self.t),
                    valinit=0,
                    )

            slider_zlim.on_changed(self.update_zlim)
            self.slider_time.on_changed(self.update_time)

            # scale x-axis same as y-axis
            ylim = self.ax_3d.get_ylim()
            xlim = self.ax_3d.get_xlim()
            ylim_delta = abs(ylim[0] - ylim[1])
            self.ax_3d.set_xlim(xlim[0] - ylim_delta/2, xlim[1] + ylim_delta/2)

            # Plot user-specified points
            for point in self.points:
                self.ax_3d.plot3D(*point, marker='.')

            # Save to output file
            if self.args.save:
                self.fig_3d.savefig(os.path.join(self.dir, "3d.png"))

        if self.args.plot:
            num_plots = len(self.config)
            fig, axs = plt.subplots(num_plots, figsize=self.figsize, squeeze=True)
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
