import json

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
from matplotlib.pyplot import ion

from Common.Utils.utils import SpeedPlan


def add_object(dic, any_object, color='#00ff00', frame=0, label=None, screens=None):
    if screens is None:
        screens = [0]
    if isinstance(screens, SpeedPlan) or isinstance(screens, int):
        screens = [screens]
    if str(frame) not in dic:
        dic[str(frame)] = {}
        for screen in screens:
            dic[str(frame)][str(screen)] = [[any_object, color, label]]
    else:
        for screen in screens:
            if str(screen) not in dic[str(frame)]:
                dic[str(frame)][str(screen)] = [[any_object, color, label]]
            else:
                dic[str(frame)][str(screen)].append([any_object, color, label])


class OccupancyViewer:
    name = 'viewer'

    def __init__(self, folder_name):
        ion()
        self.step = 0
        self.special_step = 0
        self.fig = None
        self.axes = None
        self.screen = None
        self.label_list = []
        self.polygons_dic = {}
        self.circles_dic = {}
        self.square_dic = {}
        self.lines_dic = {}
        self.point_dic = {}
        self.folder_name = folder_name

    def restart(self):
        self.step = 0
        self.label_list = []
        self.special_step = 0
        self.polygons_dic = {}
        self.circles_dic = {}
        self.lines_dic = {}
        self.point_dic = {}

    def add_polygon(self, polygon, color='#00ff00', frame=0, label=None, screens=None):
        dic = self.polygons_dic
        add_object(dic, polygon, color, frame, label, screens)

    def add_circle(self, circle, color='#000055', frame=0, label=None, screens=None):
        dic = self.circles_dic
        add_object(dic, circle, color, frame, label, screens)

    def add_square(self, square, color='#000055', frame=0, label=None, screens=None):
        dic = self.square_dic
        add_object(dic, square, color, frame, label, screens)

    def add_line(self, point_list, color='#0f0f0f', frame=0, label=None, screens=None):
        dic = self.lines_dic
        add_object(dic, point_list, color, frame, label, screens)

    def add_point(self, point, color='#0000ff', frame=0, label=None, screens=None):
        dic = self.polygons_dic
        add_object(dic, point, color, frame, label, screens)

    def draw_polygon(self, polygon, color_ext='#00ff00', color_int='#ffffff', label=None, screen=0):
        x, y = polygon.exterior.xy
        label = self.check_for_label(label)
        if label is not None:
            self.axes[screen].fill(x, y, color=color_ext, label=label)
        else:
            self.axes[screen].fill(x, y, color=color_ext)
        for poly in polygon.interiors:
            x, y = poly.xy
            self.axes[screen].fill(x, y, color=color_int)

    def draw_circle(self, circle, color='#0000ff', label=None, screen=0):
        label = self.check_for_label(label)
        if label is not None:
            c = Circle((circle[0], circle[1]), circle[2], ec=color, fc=color, alpha=1, label=label)
        else:
            c = Circle((circle[0], circle[1]), circle[2], ec=color, fc=color, alpha=1)
        self.axes[screen].add_patch(c)
        # self.axes[screen].scatter(circle[0], circle[1], circle[2], color=color, alpha=1, label=label)
        self.axes[screen].plot()

    def draw_square(self, square, color='#0000ff', label=None, screen=0):
        label = self.check_for_label(label)
        if label is not None:
            c = Rectangle((square[0] - 1.5 / 2, square[1] - 1.5 / 2), 1.5, 1.5, color=color, alpha=1, label=label)
        else:
            c = Rectangle((square[0] - 1.5 / 2, square[1] - 1.5 / 2), 1.5, 1.5, color=color, alpha=1)
        self.axes[screen].add_patch(c)
        self.axes[screen].plot()

    def draw_line(self, line, color='#0f0f0f', label=None, screen=0):
        label = self.check_for_label(label)
        try:
            line = np.array(line)
            x = line[:, 0]
            y = line[:, 1]
        except Exception as e:
            print(line[:, 0])
            raise e
        if label is not None:
            self.axes[screen].plot(x, y, color=color, alpha=1, linewidth=0.5, solid_capstyle='round', zorder=2,
                                   label=label)
        else:
            self.axes[screen].plot(x, y, color=color, alpha=1, linewidth=0.5, solid_capstyle='round', zorder=2)

    def draw_point(self, point, color='#0000ff', label=None, screen=0):
        label = self.check_for_label(label)
        # c = Circle((point[0], point[1]), 0.1, ec=color, fc=color, alpha=1, label=label, lw=0)
        # self.axes[screen].add_patch(c)
        # self.axes[screen].scatter(point[0], point[1], 0.2, color=color, alpha=1, label=label)
        if label is not None:
            self.axes[screen].plot(point[0], point[1], linewidth=0.2, color=color, alpha=1, label=label, marker='o',
                                   markersize=0.2)
        else:
            self.axes[screen].plot(point[0], point[1], linewidth=0.2, color=color, alpha=1, marker='o', markersize=0.2)

    def show_figure(self, **kwargs):
        if "title" in kwargs:
            title = kwargs["title"]
        else:
            title = None
        if "mode" in kwargs:
            mode = kwargs["mode"]
            kwargs["mode"] = str(kwargs["mode"])
        else:
            mode = None
        if "aspect" in kwargs:
            aspect = kwargs["aspect"]
        else:
            aspect = "equal"
        if "frame" in kwargs:
            frame = kwargs["frame"]
        else:
            frame = 0
        if "clean" in kwargs:
            clean = kwargs["clean"]
        else:
            clean = False
        if "step" in kwargs:
            step = kwargs["step"]
        else:
            step = True
        if "ref_time" in kwargs:
            ref_time = kwargs["ref_time"]
        else:
            ref_time = 0
        if "rec_folder" in kwargs:
            rec_folder = kwargs["rec_folder"]
        else:
            rec_folder = None
        if "world" in kwargs:
            world = kwargs["world"]
        else:
            world = None
        if 'world' in kwargs:
            del kwargs['world']
        if 'flip' in kwargs:
            flip = True
            del kwargs['flip']
        else:
            flip = False
        if 'x_lim' in kwargs:
            x_lim = kwargs["x_lim"]
        else:
            x_lim = None
        if 'y_lim' in kwargs:
            y_lim = kwargs["y_lim"]
        else:
            y_lim = None
        if 'x_label' in kwargs:
            x_label = kwargs["x_label"]
        else:
            x_label = None
        if 'y_label' in kwargs:
            y_label = kwargs["y_label"]
        else:
            y_label = None
        if 'zoom' in kwargs:
            zoom = kwargs["zoom"]
        else:
            zoom = 1
        if 'time' in kwargs:
            time = kwargs['time']
        else:
            if world is not None:
                time = world.world_time
            else:
                time = None
        if 'screen' in kwargs:
            self.screen = kwargs["screen"]
            kwargs["screen"] = str(kwargs["screen"])
        else:
            self.screen = None
        self.fig = Figure(dpi=500)
        if self.screen is not None:
            axes = self.fig.subplots(2, 2)
            self.axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
            self.fig.tight_layout(pad=3)
        else:
            self.axes = [self.fig.add_subplot(111)]
            self.screen = [0]
            self.fig.tight_layout(pad=7)
        metadata = None
        if world is not None:
            metadata = {
                "args": kwargs,
                "time": time - ref_time,
                "delay": world.world_time - (time - ref_time),
                "folder": self.folder_name
            }
        flag = 'empty'
        if str(frame) in self.polygons_dic:
            for screen in self.screen:
                if str(screen) in self.polygons_dic[str(frame)]:
                    for poly in self.polygons_dic[str(frame)][str(screen)]:
                        self.draw_polygon(poly[0], poly[1], screen=screen, label=poly[2])
                    flag = 'not empty'
        else:
            flag = 'empty'
        if str(frame) in self.lines_dic:
            for screen in self.screen:
                if str(screen) in self.lines_dic[str(frame)]:
                    for line in self.lines_dic[str(frame)][str(screen)]:
                        self.draw_line(line[0], line[1], screen=screen, label=line[2])
                    flag = 'not empty'
        if str(frame) in self.point_dic:
            for screen in self.screen:
                if str(screen) in self.point_dic[str(frame)]:
                    for point in self.point_dic[str(frame)][str(screen)]:
                        self.draw_point(point[0], point[1], screen=screen, label=point[2])
                    flag = 'not empty'
        if str(frame) in self.circles_dic:
            for screen in self.screen:
                if str(screen) in self.circles_dic[str(frame)]:
                    for cir in self.circles_dic[str(frame)][str(screen)]:
                        self.draw_circle(cir[0], cir[1], screen=screen, label=cir[2])
                    flag = 'not empty'
        if str(frame) in self.square_dic:
            for screen in self.screen:
                if str(screen) in self.square_dic[str(frame)]:
                    for square in self.square_dic[str(frame)][str(screen)]:
                        self.draw_square(square[0], square[1], screen=screen, label=square[2])
        else:
            if flag is 'empty':
                return
        for i, ax in enumerate(self.axes):
            ax.set_aspect(aspect, adjustable=None)
            if x_lim is not None:
                if i != 0:
                    x_mid = (x_lim[0] + x_lim[1]) / 2
                    x_dif = ((x_lim[1] - x_lim[0]) / 2) / zoom
                    new_x_lim = [x_mid - x_dif, x_mid + x_dif]
                    ax.set_xlim(new_x_lim)
                else:
                    ax.set_xlim(x_lim)
            if y_lim is not None:
                if i != 0:
                    y_mid = (y_lim[0] + y_lim[1]) / 2
                    y_dif = ((y_lim[1] - y_lim[0]) / 2) / zoom
                    new_y_lim = [y_mid - y_dif, y_mid + y_dif]
                    ax.set_ylim(new_y_lim)
                else:
                    ax.set_ylim(y_lim)
            if flip:
                # ax.invert_yaxis()
                ax.invert_xaxis()
            if i == 0:
                if mode is not None:
                    ax.set_title(mode)
                else:
                    ax.set_title("no mode selected")
            else:
                ax.set_title("mode : " + str(i))
            ax.legend(
                loc='upper left',
                bbox_to_anchor=(1.05, 0, 0.7, 1),
                mode='expand',
                frameon=False,
                borderaxespad=0.,
                fontsize='x-small',
                handlelength=0.5
            )
            if x_label is not None:
                ax.set_xlabel(x_label, fontsize='x-small')
            if y_label is not None:
                ax.set_ylabel(y_label, fontsize='x-small')
            ax.tick_params(labelsize='x-small')
        if title is not None:
            self.fig.suptitle(title, fontsize='x-large')
        if step:
            self.fig.savefig(rec_folder + "/" + self.folder_name + '/' + str(self.step) + '.png', format='png')
            if metadata is not None:
                with open(rec_folder + '/metadata/' + self.folder_name + '_' + str(self.step) + '.json', 'w') \
                        as json_file:
                    json.dump(metadata, json_file)
            self.step += 1
            self.special_step = 0
        else:
            self.fig.savefig(
                rec_folder + '/' + self.folder_name + '/' + str(self.step + 1) + '_'
                + str(self.special_step) + '.png', format='png')
            if metadata is not None:
                with open(
                        rec_folder + '/metadata/' + self.folder_name + '_' + str(self.step + 1) + '_' +
                        str(self.special_step)
                        + '.json',
                        'w'
                ) as json_file:
                    json.dump(metadata, json_file)
            self.special_step += 1
        for ax in self.axes:
            ax.clear()
        if clean:
            clean_dic_from_old_frame(self.circles_dic, frame)
            clean_dic_from_old_frame(self.polygons_dic, frame)
            clean_dic_from_old_frame(self.lines_dic, frame)
            clean_dic_from_old_frame(self.point_dic, frame)
        else:
            if str(frame) in self.circles_dic:
                del self.circles_dic[str(frame)]
            if str(frame) in self.polygons_dic:
                del self.polygons_dic[str(frame)]
            if str(frame) in self.lines_dic:
                del self.lines_dic[str(frame)]
            if str(frame) in self.point_dic:
                del self.point_dic[str(frame)]
        self.label_list = []

    def check_for_label(self, label):
        if label in self.label_list:
            label = None
        else:
            self.label_list.append(label)
        return label


def clean_dic_from_old_frame(dic, last_frame):
    frame2del = []
    for frame_index in dic:
        if int(frame_index) <= last_frame:
            frame2del.append(frame_index)
    for frame in frame2del:
        del dic[frame]
