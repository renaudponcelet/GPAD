import json
import math

import numpy as np
import shapely.geometry
from matplotlib.pyplot import ion, Circle, Figure, Rectangle
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, LinearRing, MultiPolygon
from shapely.ops import unary_union


class PathToShort(Exception):
    pass


class SpeedPlan:
    possible_value = [None, "brake", "keep_speed", "speed_up"]

    def __init__(self, value):
        if isinstance(value, str) and value in SpeedPlan.possible_value:
            self.value = SpeedPlan.possible_value.index(value)
        elif isinstance(value, int):
            _ = SpeedPlan.possible_value[value]
            self.value = value

    def __int__(self):
        return self.value

    def __str__(self):
        return SpeedPlan.possible_value[self.value]

    def __eq__(self, other):
        if isinstance(other, int):
            return self.value == other
        elif isinstance(other, str):
            return str(self.value) == other
        else:
            return self.value == other.value

    def __lt__(self, other):
        if isinstance(other, int):
            return self.value < other
        elif isinstance(other, str):
            return self < SpeedPlan(other)
        else:
            return self.value < other.value


def find_nearest_vector(array, value):
    if len(array) <= 1:
        return 0, 0
    value = np.copy(np.asarray(value))
    idx = (np.linalg.norm(np.add(array, - value), axis=1)).argsort(axis=0)
    if idx[0] - 1 >= 0 and idx[0] + 1 < len(array):
        v0 = np.add(array[idx[0] - 1], - array[idx[0]])
        v1 = np.add(array[idx[0] + 1], - array[idx[0]])
        vv = np.add(value, - array[idx[0]])
        if np.dot(vv, v0) > np.dot(vv, v1):
            return idx[0], idx[0] - 1
        else:
            return idx[0] + 1, idx[0]
    elif idx[0] - 1 < 0:
        return 1, 0
    else:
        return len(array) - 1, len(array) - 2


def find_nearest(array, value):
    idx = (np.abs(np.add(array, - value))).argsort(axis=0)
    return idx[0]


def dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def dist_3d(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)


def norm_x_y(vector):
    return math.sqrt(vector.x ** 2 + vector.y ** 2)


def first_order(input_value, tau, t, last_output):
    e = math.exp(- t / tau)
    return e * last_output + (1 - e) * input_value


def get_point_on_seg(a, b, distance):
    ab = np.add(b, - a)
    point = np.add(a, distance * (ab / np.linalg.norm(ab)))
    return np.array([point[0], point[1]]).round(decimals=2)


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


def get_index_offset(global_path_point, vehicle_location, world, distance=None):
    if distance is None:
        distance = world.visual_horizon
    indexes = find_nearest_vector(global_path_point, vehicle_location)
    dot = (global_path_point[max(indexes)] - global_path_point[max(indexes) - 1]).dot(
        vehicle_location - global_path_point[max(indexes) - 1])
    if dot >= 0:
        index = max(indexes)
    else:
        index = max(indexes) - 1
    index_offset = int(distance // world.global_path_interval)
    return index, index_offset


def path2rs(path, pos, yaw, speed_plan, time_step,
            dist_step, u_max, wheel_spacing, L, horizon,
            nb_circles=0, ego_offset=0, sort=False, index_limit=None,
            flat=False):
    #  path = [x, y]
    if sort and flat:
        raise Warning(
            'Use path2rs with sort mode and flat mode is useless for now but it can be implemented if it\'s needed'
        )
    path = get_realistic_path(path, pos, yaw, dist_step, u_max, wheel_spacing, L, horizon, index_limit)
    if len(path) < 3:
        raise PathToShort
    path_dist = []
    ego_dist = []
    for index in range(len(path) - 1):
        path_dist.append(dist(path[index + 1], path[index]))
    for index in range(len(speed_plan) - 1):
        ego_dist.append((speed_plan[index] + speed_plan[index + 1]) / 2 * time_step)
    if flat:
        rs = np.array([[path[0][0], path[0][1], yaw]])
    else:
        rs = np.array([[0, path[0][0], path[0][1], yaw]])
    index_ego = 0
    index_path = 0
    path_dist_rest = ego_dist[index_ego]
    sorting_value = 0
    last_point = None
    for _ in range(1000):
        if path_dist_rest < path_dist[index_path]:
            point_on_seg = get_point_on_seg(
                path[index_path], path[index_path + 1], path_dist_rest)
            path_dir = np.add(path[index_path + 1], - path[index_path])
            if sort:
                if last_point is None:
                    last_point = path[index_path]
                sorting_value += math.sqrt(
                    (point_on_seg[0] - last_point[0]) ** 2 + (point_on_seg[1] - last_point[1]) ** 2
                )
            if flat:
                new_yaw = np.arctan2(path_dir[1], path_dir[0])
                if new_yaw <= 0:
                    new_yaw += 2 * math.pi
                if new_yaw > 2 * math.pi:
                    new_yaw -= 2 * math.pi
                rs = np.vstack(
                    (
                        rs,
                        np.array([np.array([point_on_seg[0], point_on_seg[1], new_yaw])])
                    )
                )
            else:
                new_yaw = np.arctan2(path_dir[1], path_dir[0])
                if new_yaw <= 0:
                    new_yaw += 2 * math.pi
                if new_yaw > 2 * math.pi:
                    new_yaw = 2 * math.pi
                rs = np.vstack(
                    (
                        rs,
                        np.array(
                            [np.array(
                                [sorting_value, point_on_seg[0], point_on_seg[1], new_yaw]
                            )]
                        )
                    )
                )
            if index_limit is not None:
                if len(rs) >= index_limit:
                    break
            last_point = point_on_seg
            index_ego += 1
            if index_ego >= len(speed_plan) - 1:
                break
            path_dist_rest += ego_dist[index_ego]
        else:
            index_path += 1
            if index_path >= len(path) - 1:
                break
            path_dist_rest -= path_dist[index_path]
    if flat:
        return rs
    ran = range(-nb_circles // 2 + 1, nb_circles // 2 + 1)
    if nb_circles % 2 == 0:
        b_line_factors = [r - 0.5 for r in ran]
    else:
        b_line_factors = list(ran)
    based_line = np.array([[0, ego_offset * f] for f in b_line_factors])
    out = []
    for i, obj in enumerate(rs):
        x = obj.dot(np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))
        theta = obj[3] - math.pi / 2
        obs_circles = []
        rot = np.array([
            [math.cos(theta), math.sin(theta)],
            [- math.sin(theta), math.cos(theta)]
        ]).round(decimals=2)
        rotated_line = np.array([point.dot(rot) for point in based_line])
        trans = x
        res = np.add(rotated_line, trans)
        obs_circles.append(np.array(
            [obj] + [[0, row[0], row[1], 0] for row in res]
        ).round(decimals=2))
        out.append(obs_circles)
    return out


def get_key(slice_id, group_id):
    return str(slice_id) + "-" + str(group_id)


def get_projected_polygons(csg, projection_plan='xy'):
    polygons = []
    poly_index = {}
    for i, csg_slice in enumerate(csg):
        if i == len(csg) - 1:
            continue
        if isinstance(csg_slice, float):
            continue
        next_slice = csg[i + 1]
        for j, group in enumerate(csg_slice):
            # x = [tri, x, y, theta]
            if isinstance(group, np.ndarray) and group.shape[1] == 4:
                group_in_poly = False
                max_group_j = np.amax(group, axis=0)[0]
                min_group_j = np.amin(group, axis=0)[0]
                for k, nextSliceGroup in enumerate(next_slice):
                    max_group_k = np.amax(nextSliceGroup, axis=0)[0]
                    min_group_k = np.amin(nextSliceGroup, axis=0)[0]
                    if max_group_k >= min_group_j <= min_group_k or max_group_k >= max_group_j <= min_group_k:
                        group_key = get_key(i, j)
                        if group_key in poly_index:
                            poly_key = poly_index[group_key]
                            last_poly = polygons[poly_key]
                            if projection_plan == 'xy':
                                points2add = nextSliceGroup[:, 1:3]
                            elif projection_plan == 'ts':
                                #  beware time is int and must be multiply by a time step
                                points2add = np.insert(
                                    [nextSliceGroup[:, 0].reshape(len(nextSliceGroup), 1)],
                                    0,
                                    i + 1,
                                    axis=2
                                )[0]
                            else:
                                raise Exception("Wrong projection plan")
                            new_poly = np.concatenate((last_poly, points2add), axis=0)
                            polygons[poly_key] = new_poly
                            poly_index[get_key(i + 1, k)] = poly_key
                        else:
                            if projection_plan == 'xy':
                                points2add = group[:, 1:3]
                                points2add_next_slice = nextSliceGroup[:, 1:3]
                            elif projection_plan == 'ts':
                                #  beware time is int and must be multiply by a time step
                                points2add = np.insert(
                                    [group[:, 0].reshape(len(group), 1)],
                                    0,
                                    i,
                                    axis=2
                                )[0]
                                points2add_next_slice = np.insert(
                                    [nextSliceGroup[:, 0].reshape(len(nextSliceGroup), 1)],
                                    0,
                                    i + 1,
                                    axis=2
                                )[0]
                            else:
                                raise Exception("Wrong projection plan")
                            new_poly = np.concatenate((points2add, points2add_next_slice), axis=0)
                            poly_key = len(polygons)
                            polygons.append(new_poly)
                            poly_index[get_key(i, j)] = poly_key
                            poly_index[get_key(i + 1, k)] = poly_key
                        group_in_poly = True
                    else:
                        continue
                if not group_in_poly:
                    group_key = get_key(i, j)
                    if group_key not in poly_index:
                        poly_key = len(polygons)
                        polygons.append(group[:, 1:3])
                        poly_index[group_key] = poly_key
            else:
                raise Exception("Wrong type for group")
    return polygons


def get_lazy_path(pos, global_path_point, world):
    loc = np.array([pos.x, pos.y]).round(decimals=2)
    idx, index_offset = get_index_offset(global_path_point, loc, world)
    return global_path_point[idx:idx + index_offset]


def find_paths(graph, start, end, path=None):
    if path is None:
        path = []
    path = path + [start]
    if start == end:
        return [path]
    if start not in graph:
        return []
    paths = []
    for node in graph[start]['children']:
        if node not in path:
            new_paths = find_paths(graph, node, end, path)
            for new_path in new_paths:
                paths.append(new_path)
    return paths


def get_visible_exteriors_from_polys(point, polys, dist_horizon, world, screens=None):
    point_shapely = Point(point)
    lines = []
    visible_segments = []
    for poly in polys:
        if poly.distance(point_shapely) <= dist_horizon:
            linear_ring = LinearRing()
            if poly.contains(point_shapely):
                if world.display and screens is not None:
                    world.occupancy_viewer_ris.add_polygon(poly, color='#ff0000',
                                                           screens=screens,
                                                           frame=world.recorder.current_frame)
                distance = poly.exterior.distance(point_shapely)
                for interior in poly.interiors:
                    dist_temp = interior.distance(point_shapely)
                    if dist_temp < distance:
                        distance = dist_temp
                poly = poly.buffer(- (distance + world.tolerance))
                if poly.contains(point_shapely):
                    raise Exception("negative buffer is not sufficient..")
                if isinstance(poly, Polygon):
                    if poly.is_empty:
                        print("poly is empty")
                        continue
                        # raise Exception("poly is empty")
                    if world.display and screens is not None:
                        world.occupancy_viewer_ris.add_polygon(poly, color='#ff00ff',
                                                               screens=screens,
                                                               frame=world.recorder.current_frame)
                    selected_poly = poly
                elif isinstance(poly, MultiPolygon):
                    min_dist = float('inf')
                    selected_poly = None
                    for sub_poly in poly.geoms:
                        if world.display and screens is not None:
                            world.occupancy_viewer_ris.add_polygon(sub_poly, color='#ff00ff',
                                                                   screens=screens,
                                                                   frame=world.recorder.current_frame)
                        dist_to_poly = sub_poly.distance(point_shapely)
                        if dist_to_poly < min_dist:
                            selected_poly = sub_poly
                            min_dist = dist_to_poly
                else:
                    raise Exception("not a poly")
                flag = False
                for interior in selected_poly.interiors:
                    if Polygon(interior).contains(point_shapely):
                        linear_ring = interior
                        flag = True
                if not flag:
                    linear_ring = selected_poly.exterior
            else:
                flag = False
                for interior in poly.interiors:
                    if Polygon(interior).contains(point_shapely):
                        linear_ring = interior
                        flag = True
                        break
                if not flag:
                    linear_ring = poly.exterior
            if linear_ring is None:
                raise Exception("None line")
            lines.append(linear_ring)
    if world.display and screens is not None:
        for line in lines:
            world.occupancy_viewer_ris.add_line(list(line.coords), color='#ff00ff',
                                                screens=screens,
                                                frame=world.recorder.current_frame)
    visible_line = is_visible(point_shapely, lines, dist_horizon)
    for s in np.arange(0, visible_line.length, world.global_path_interval):
        p0 = np.array(visible_line.interpolate(s - world.global_path_interval).coords[0])
        p1 = np.array(visible_line.interpolate(s).coords[0])
        p2 = np.array(visible_line.interpolate(s + world.global_path_interval).coords[0])
        vector1 = np.add(p1, - p0)
        normal1 = np.array([- vector1[1], vector1[0]])
        vector2 = np.add(p2, - p1)
        normal2 = np.array([- vector2[1], vector2[0]])
        normal_moy = np.add(normal1, normal2)
        normal_moy /= np.linalg.norm(normal_moy)
        visible_segments.append((p1, normal_moy))
    return np.array(visible_segments)


def is_visible(point, lines, radius=100):
    sweep_res = 5
    line = LineString([(point.x, point.y), (point.x, point.y + radius)])
    all_input_lines = unary_union(lines)

    perimeter = []
    # traverse each radial sweep line and check for intersection with input lines
    for radial_line in [affinity.rotate(line, i, (point.x, point.y)) for i in range(0, 360, sweep_res)]:
        inter = radial_line.intersection(all_input_lines)

        if inter.type == "MultiPoint":
            # radial line intersects at multiple points
            inter_dict = {}
            for inter_pt in inter:
                inter_dict[point.distance(inter_pt)] = inter_pt
            # save the nearest intersected point to the sweep centre point
            perimeter.append(inter_dict[min(inter_dict.keys())])

        if inter.type == "Point":
            # radial line intersects at one point only
            perimeter.append(inter)

        if inter.type == "GeometryCollection":
            # radial line doesn't intersect, so add the end point of the line
            perimeter.append(Point(radial_line.coords[1]))

    # combine the nearest perimeter points into one geometry
    return LinearRing([np.array(p) for p in perimeter]).simplify(0.1)


def get_diff_angle(u, d, wheel_spacing):
    return math.sin(u) * d / wheel_spacing


def get_wheel_angle(delta_theta, d, wheel_spacing):
    return np.arcsin(wheel_spacing * delta_theta / d)


def get_realistic_path(path, start_loc, start_heading, dist_step, u_max, wheel_spacing, L, horizon, max_index=None):
    if len(path) == 0:
        return np.array([])
    if start_heading <= 0:
        start_heading += 2 * math.pi
    if start_heading > 2 * math.pi:
        start_heading -= 2 * math.pi
    path_point = path[:, :2]
    realistic_path = [[start_loc[0], start_loc[1], start_heading]]
    last_point = start_loc
    last_heading = start_heading
    flag = False
    nb_step = int(horizon / dist_step)
    previous_index = 0
    for i in range(nb_step):
        u_0 = np.array([math.cos(last_heading), math.sin(last_heading)]).round(decimals=2)
        # the forward point distance smooth the path_point, usually we take wheel spacing but i set to other value
        # to debug realistic_path doing strange things
        forward_point = np.add(last_point, L * u_0)
        indexes = find_nearest_vector(path_point, forward_point)
        path_vector = np.add(path_point[max(indexes)], - path_point[min(indexes)])
        forward_vector = np.add(forward_point, - path_point[min(indexes)])
        dot_product = np.dot(forward_vector, path_vector)
        if dot_product >= 0:
            target_point = path_point[max(indexes)]
            if max(indexes) < previous_index:
                indexes = [previous_index]
            previous_index = max(indexes)
        else:
            target_point = path_point[min(indexes)]
            if min(indexes) < previous_index:
                indexes = [previous_index]
            previous_index = min(indexes)
        target_vector = np.add(target_point, - np.array(last_point))
        target_heading = np.arctan2(target_vector[1], target_vector[0])
        if target_heading <= 0:
            target_heading += 2 * math.pi
        if target_heading > 2 * math.pi:
            target_heading -= 2 * math.pi
        if max_index is not None and min(indexes) >= max_index:
            last_point = target_point
            last_heading = target_heading
        else:
            diff_heading = target_heading - last_heading
            if abs(diff_heading) > math.pi:
                if diff_heading < - math.pi:
                    diff_heading += 2 * math.pi
                else:
                    diff_heading -= 2 * math.pi
                if abs(diff_heading) > math.pi:
                    raise Exception("bad angle")
            max_diff_heading = get_diff_angle(u_max, dist_step, wheel_spacing)
            if abs(diff_heading) > max_diff_heading:
                diff_heading = np.sign(diff_heading) * max_diff_heading
                u = np.sign(diff_heading) * u_max
            else:
                u = get_wheel_angle(diff_heading, dist_step, wheel_spacing)
            u_theta = np.array([math.cos(last_heading + u), math.sin(last_heading + u)]).round(decimals=2)
            last_heading += diff_heading
            if last_heading <= 0:
                last_heading += 2 * math.pi
            if last_heading > 2 * math.pi:
                last_heading -= 2 * math.pi
            u_1 = np.array([math.cos(last_heading), math.sin(last_heading)]).round(decimals=2)
            delta_loc = np.add(dist_step * u_theta, - wheel_spacing / 2 * np.add(u_1, - u_0))
            last_point = np.add(last_point, delta_loc)
        realistic_path.append([last_point[0], last_point[1], last_heading])
        if max(indexes) >= len(path_point) - 1:
            flag = True
            if len(realistic_path) == 2:
                debug_path = np.array(realistic_path)
                debug_path_vector = np.add(debug_path[1], - debug_path[0])
                forward_point = [debug_path[1, 0] + debug_path_vector[0],
                                 debug_path[1, 1] + debug_path_vector[1],
                                 debug_path[1, 2]]
                realistic_path.append(forward_point)
            break
    if not flag:
        print(path_point)
    if len(realistic_path) == 2:
        debug_path = np.array(realistic_path)
        debug_path_vector = np.add(debug_path[1], - debug_path[0])
        forward_point = [debug_path[1, 0] + debug_path_vector[0],
                         debug_path[1, 1] + debug_path_vector[1],
                         debug_path[1, 2]]
        realistic_path.append(forward_point)
    final_path = np.array(realistic_path)
    if len(final_path) < 3:
        raise PathToShort
    return final_path


def clean_obstacles(obstacles, buffer_size=0, tolerance=0):
    # Merge superposed obstacles
    obs = None
    try:
        polygons = []
        for obs in obstacles:
            poly = obs.buffer(buffer_size)
            polygons.append(poly.simplify(tolerance, preserve_topology=True))
    except Exception as e:
        print(e)
        try:
            polygons = []
            for obs in obstacles:
                poly = Polygon(np.unique(obs, axis=0)).buffer(buffer_size)
                polygons.append(poly.simplify(tolerance, preserve_topology=True))
        except Exception as e:
            with open('logs.json', 'w') as f:
                json.dump({"obs": list(obs.exterior.coords)}, f)
            raise e
    merged = unary_union(polygons)
    if isinstance(merged, shapely.geometry.multipolygon.MultiPolygon):
        return list(merged.geoms)
    elif isinstance(merged, shapely.geometry.polygon.Polygon):
        return [merged]
    else:
        return []


def get_intersecting_poly(iz, virtual_iz, line):
    intersects_poly = []
    for poly in virtual_iz:
        if poly.intersects(line):
            intersects_poly.append(poly)
    if len(intersects_poly) > 0:
        return intersects_poly, True
    for poly in iz:
        if poly.intersects(line):
            intersects_poly.append(poly)
    return intersects_poly, False


def get_cross_max_point(intersects_poly, origin, eps=0.3):
    points = None
    for poly in intersects_poly:
        if points is None:
            points = np.array(poly.exterior.coords)
        else:
            points = np.vstack((points, np.array(poly.exterior.coords)))
    poly = Polygon(
        list(point) for point in points
    ).convex_hull.buffer(eps, cap_style=2, join_style=2).simplify(eps).convex_hull
    cross_max = - float('inf')
    passing_point = None
    for point in poly.exterior.coords:
        point = np.array(point) - origin
        if point[0] == 0:
            continue
        sloop = point[1] / point[0]
        if sloop > cross_max:
            passing_point = point
            cross_max = sloop
    return passing_point


def get_cross_min_point(intersects_poly, origin, eps=0.3, cross_max=None):
    points = None
    for poly in intersects_poly:
        if not isinstance(poly, Polygon):
            if isinstance(poly, LineString):
                poly = poly.buffer(0.1)
            else:
                raise ValueError
        if points is None:
            points = np.array(poly.exterior.coords)
        else:
            points = np.vstack((points, np.array(poly.exterior.coords)))
    poly = Polygon(
        list(point) for point in points
    ).convex_hull.buffer(eps, cap_style=2, join_style=2).simplify(eps).convex_hull
    if cross_max is None:
        cross_min = float('inf')
    else:
        cross_min = cross_max
    passing_point = None
    for point in poly.exterior.coords:
        point = np.add(np.array(point), -origin)
        if point[0] <= 0:
            continue
        sloop = point[1] / point[0]
        if sloop < cross_min:
            passing_point = point
            cross_min = sloop
    return passing_point


def check_collision(s, iz, virtual_iz, time_line):
    for poly in virtual_iz:
        for point in s:
            if poly.contains(Point((point[0], point[1]))):
                return [poly], True, np.array(point), 'under'
    for poly in iz:
        for point in s:
            if poly.contains(Point((point[0], point[1]))):
                last_action = False
                for action in time_line:
                    if not last_action and point[0] >= action[0]:
                        last_action = True
                        continue
                    if last_action:
                        if action[1] == 'over':
                            return action[2], False, np.array(point), 'over'
                        else:
                            return [poly], False, np.array(point), 'under'
                if last_action:
                    return [poly], False, np.array(point), 'under'
    return None, False, None, 'under'


def get_s_from_speed_plan(speed_plan):
    s = [[0, 0]]
    current_time = None
    current_speed = None
    flag = False
    for speed in speed_plan:
        if flag is False:
            current_time = speed[0]
            current_speed = speed[1]
            flag = True
        else:
            s.append(
                [speed[0], s[-1][1] + current_speed * (speed[0] - current_time)]
            )
            current_time = speed[0]
            current_speed = speed[1]
    return s


def get_secure_dist(u_max, dist_step, wheel_spacing, e_y):
    last_point = np.array([0, 0])
    last_heading = 0
    max_diff_heading = get_diff_angle(u_max, dist_step, wheel_spacing)
    while last_point[1] < e_y:
        u_0 = np.array([math.cos(last_heading), math.sin(last_heading)]).round(decimals=2)
        u_theta = np.array([math.cos(last_heading + u_max), math.sin(last_heading + u_max)]).round(decimals=2)
        last_heading += max_diff_heading
        u_1 = np.array([math.cos(last_heading), math.sin(last_heading)]).round(decimals=2)
        delta_loc = np.add(dist_step * u_theta, - wheel_spacing / 2 * np.add(u_1, - u_0))
        last_point = np.add(last_point, delta_loc)
    return last_point[0]


def super_dic(s_dic, dic):
    out_dic = s_dic
    for key in dic:
        out_dic[key] = dic[key]
    return out_dic
