import math

import carla
import numpy as np
from PythonAPI.GPAD.Common.Utils.carla_utils import get_realistic_path
from shapely.geometry import LineString, MultiLineString, Polygon, Point


class ReachableSetGenerator:
    name = "class to generate reachable sets"

    def __init__(self, world, u_max, wheel_spacing):
        self.world = world
        self.rs = None
        self.static_ris = None
        self.u_max = np.deg2rad(u_max)
        self.wheel_spacing = wheel_spacing

    def run(self, static_path, width, speed_plan, speed_profile=None, blocked=False, nb_step=None):
        self.rs, self.static_ris = self.set_rs_around_path(static_path, width, speed_plan, speed_profile=speed_profile,
                                                           blocked=blocked, nb_step=nb_step)
        return self.rs, self.static_ris

    def set_rs_around_path(self, path, width, speed_plan, speed_profile=None, blocked=False, nb_step=None):
        if nb_step is None:
            nb_step = self.world.nb_step
        shapely_path = LineString([list(point[:2]) for point in path]).simplify(self.world.tolerance)
        width_tab = width * (np.arange(
            2 * (self.world.nb_paths + 1) + 1
        ) - (self.world.nb_paths + 1)) / self.world.nb_paths
        targets_maneuver = []
        for width in width_tab:
            if width == 0:
                targets_maneuver.append(shapely_path)
            else:
                try:
                    parallel_offset = shapely_path.parallel_offset(width, 'left')
                except Exception:
                    parallel_offset = None
                if isinstance(parallel_offset, LineString):
                    if width < 0:
                        targets_maneuver.append(LineString(parallel_offset.coords[::-1]))
                    else:
                        targets_maneuver.append(parallel_offset)
                elif isinstance(parallel_offset, MultiLineString):
                    lines_concatenated = None
                    for line in parallel_offset:
                        if lines_concatenated is None:
                            lines_concatenated = line.coords
                        else:
                            lines_concatenated = np.concatenate((lines_concatenated, line.coords), axis=0)
                    print(lines_concatenated)
                    if width < 0:
                        targets_maneuver.append(LineString(lines_concatenated[::-1]))
                    else:
                        targets_maneuver.append(LineString(lines_concatenated))
                else:
                    targets_maneuver.append(LineString())
        sub_paths = []
        dist_step = self.world.vehicle_speed_limit * self.world.time_step_res
        if dist_step < self.world.global_path_interval:
            dist_step = self.world.global_path_interval
        ris = []
        extreme_lines = []
        for i, target in enumerate(targets_maneuver):
            s_tab = np.arange(0, target.length, dist_step)
            target_path = []
            for s in s_tab:
                target_path.append(np.array(target.interpolate(s)))
            realistic_path = get_realistic_path(
                np.array(target_path),
                path[0, :2],
                path[0, 2],
                dist_step,
                self.u_max / 3,
                self.wheel_spacing,
                self.world.scenario_data["L"],
                self.world.visual_horizon
            )
            # if speed_profile == 'brake':
            #     screen = 1
            # elif speed_profile == 'keep_speed':
            #     screen = 2
            # else:
            #     screen = 3
            # self.world.occupancy_viewer_ris.add_line(np.array(target_path)[:, :2],
            #                                          frame=self.world.recorder.current_frame,
            #                                          color='#550000',
            #                                          screens=screen)
            if i == 0 or i == 2 * (self.world.nb_paths + 1):
                line = []
                for j, point in enumerate(realistic_path):
                    line.append(point)
                extreme_lines.append(line)
            else:
                sub_paths.append(
                    LineString([list(point[:2]) for point in realistic_path])
                )
        poly = []
        for j in range(2):
            line = extreme_lines[j][::pow(-1, j + 1)]
            for point in line:
                poly.append(point)
        start_vector_norm = 0
        count = 0
        while start_vector_norm == 0:
            if len(path) > (2 + count):
                start_vector = np.add(path[2 + count], - path[0])[:2]
            elif len(path) == 2 + count:
                start_vector = np.add(path[1 + count], - path[0])[:2]
            else:
                return None, None
            start_vector_norm = np.linalg.norm(start_vector)
            if start_vector_norm > 0:
                start_vector /= start_vector_norm
            else:
                count += 1
        normal_vector = np.array([- start_vector[1], start_vector[0]])
        start_vector *= self.world.global_path_interval
        behind_point = np.add(path[0, :2], - width * start_vector)
        l_limit = behind_point + 2 * width * normal_vector
        poly.append(l_limit)
        r_limit = behind_point - 2 * width * normal_vector
        poly.append(r_limit)
        poly.append(poly[0])
        point_shapely = Point(path[0][:2]).buffer(10)
        poly_shapely = Polygon([list(point[:2]) for point in poly])
        if poly_shapely.is_valid:
            ris.append(poly_shapely.difference(point_shapely))
        else:
            poly_shapely = poly_shapely.buffer(0)
            if poly_shapely.is_valid:
                ris.append(poly_shapely.difference(point_shapely))
            else:
                raise Exception("poly_shapely is not valid")
        free_path_flag = np.full(2 * self.world.nb_paths + 1, True, dtype=bool)
        if self.world.planner.kool_down or str(self.world.occupancy_mapper.traffic_light_state) == 'Red':
            free_path_flag[:int((2 * self.world.nb_paths + 1) / 3)] = False
        x = [[]]
        x[0].append([0, path[0, 0], path[0, 1], path[0, 2]])
        for t in range(nb_step - 1):
            x.append([])
        s = 0
        prev_x = []
        for i in range(2 * self.world.nb_paths + 1):
            prev_x.append([path[0, 0], path[0, 1]])
        circles = self.world.occupancy_mapper.signal_obstacles[str(
            self.world.recorder.current_frame)]
        min_dist = float('inf')
        lines = []
        for _ in free_path_flag:
            lines.append([])
        for t in range(nb_step - 1):
            s += speed_plan[t] * self.world.time_step_res
            print(t, free_path_flag)
            for i, flag in enumerate(free_path_flag):
                if sub_paths[i].is_empty:
                    free_path_flag[i] = False
                    flag = False
                new_x = np.array(sub_paths[i].interpolate(s))
                if flag:
                    if (str(self.world.recorder.current_frame) in self.world.occupancy_mapper.signal_obstacles
                            and circles is not None and len(circles) > t and circles[t] is not None):
                        for circle in circles[t]:
                            if circle is None:
                                continue
                            dist = math.sqrt((new_x[0] - circle[0]) ** 2 + (new_x[1] - circle[1]) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                            if dist < circle[2] + \
                                    2 * self.world.ego_radius:
                                for j in range(len(free_path_flag)):
                                    free_path_flag[j] = False
                                continue
                    new_x_waypoint = self.world.map.get_waypoint(carla.Location(x=new_x[0], y=new_x[1]))
                    if not blocked and new_x_waypoint.is_junction and i < int(len(free_path_flag) / 2) - 1:
                        free_path_flag[i] = False
                        continue
                    if self.world.occupancy_mapper.in_map([new_x], eps=self.world.vehicle.bounding_box.extent.y,
                                                          screen=speed_plan):
                        if t * self.world.time_step_res > self.world.time_horizon / 2:
                            lines[i].append(new_x)
                        free_path_flag[i] = False
                    else:
                        theta = np.arctan2(prev_x[i][1] - new_x[1], prev_x[i][0] - new_x[1])
                        if theta <= 0:
                            theta += 2 * math.pi
                        if theta > 2 * math.pi:
                            theta -= 2 * math.pi
                        if t != 0:
                            for j, value in enumerate(x[t]):
                                if value[0] == i:
                                    x[t][j][3] = theta
                                    break
                        prev_x[i] = new_x
                        x[t + 1].append([i, new_x[0], new_x[1], 0])
                else:
                    if t * self.world.time_step_res > self.world.time_horizon / 2:
                        lines[i].append(new_x)
        for i, flag in enumerate(free_path_flag):
            if len(lines[i]) == 1:
                ris.append(Point(lines[i][0][:2]))
            else:
                ris.append(LineString([list(point[:2]) for point in lines[i]]))
        print(min_dist)
        vertices = np.full(len(x), 0., dtype=object)
        for i in range(len(x)):
            vertices[i] = np.array(x[i])
        if self.world.nb_circles % 2 == 0:
            ran = range(-self.world.nb_circles // 2 + 1, self.world.nb_circles // 2 + 1)
            b_line_factors = [r - 0.5 for r in ran]
        else:
            ran = range(-self.world.nb_circles // 2, self.world.nb_circles // 2 + 1)
            b_line_factors = list(ran)
        based_line = np.array([[self.world.ego_offset * f, 0] for f in b_line_factors])
        for i, obj in enumerate(vertices):
            if len(obj) == 0:
                continue
            x = obj.dot(np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))
            thetas = obj[:, 3]
            obs_circles = []
            for j, theta in enumerate(thetas):
                rot = np.array([
                    [math.cos(theta), math.sin(theta)],
                    [- math.sin(theta), math.cos(theta)]
                ])
                rotated_line = np.array([point.dot(rot) for point in based_line])
                trans = x[j]
                res = np.add(rotated_line, trans)
                obs_circles.append(np.array(
                    [obj[j]] + [[0, row[0], row[1], 0] for row in res]
                ))
            vertices[i] = np.array(obs_circles)
            if self.world.display:
                if speed_profile == 'brake':
                    screen = 1
                elif speed_profile == 'keep_speed':
                    screen = 2
                elif speed_profile == 'speed_up':
                    screen = 3
                else:
                    screen = 0
                if i == 0:
                    for circles in obs_circles:
                        for c, circle in enumerate(circles):
                            if c == 0:
                                continue
                            else:
                                self.world.occupancy_viewer_ris.add_circle(
                                    [circle[1], circle[2], self.world.ego_radius],
                                    frame=self.world.recorder.current_frame,
                                    color='#550000',
                                    screens=screen,
                                    label='ego vehicle'
                                )
        return vertices, ris
