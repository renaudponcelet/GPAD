import math

import numpy as np
from ...Common.Utils.utils import get_projected_polygons, find_nearest_vector, clean_obstacles, \
    get_intersecting_poly, get_cross_max_point, get_cross_min_point, check_collision, get_s_from_speed_plan, \
    get_out_path
from shapely.geometry import Point, LineString, Polygon


class SGSPA:
    name = "SGSPA class for Carla simulator"

    def __init__(self, world):
        self.world = world
        self.emergency_mode = False
        self.__max_acc = None
        self.__time_gap = None
        self.path = None
        self.safe_time_gap = world.time_horizon * 0.05
        self.last_critical_poly = None

    @property
    def max_acc(self):
        if self.emergency_mode:
            self.max_acc = self.world.max_acc_emergency
        else:
            self.max_acc = self.world.max_acc
        return self.__max_acc

    @max_acc.setter
    def max_acc(self, acc):
        self.__max_acc = acc

    @property
    def time_gap(self):
        self.time_gap = self.world.time_gap
        for junction in self.world.occupancy_mapper.static_conditional_virtual_obstacles[str(
                self.world.recorder.current_frame)]:
            if junction.contains(Point((self.world.vehicle.get_location().x, self.world.vehicle.get_location().y))):
                if self.world.cross_security:
                    self.time_gap = self.safe_time_gap
                if self.world.time_in_junction['first_time'] == 0:
                    self.world.time_in_junction['first_time'] = self.world.world_time
                else:
                    self.world.time_in_junction['last_time'] = self.world.world_time
                break
        return self.__time_gap

    @time_gap.setter
    def time_gap(self, time):
        self.__time_gap = time

    def get_speed(self, path):
        self.emergency_mode = False
        diagonal_speed, path_length = self.world.occupancy_mapper.update_rm(path, self.world.vehicle.get_location(),
                                                                            self)
        iz = self.compute_iz()
        virtual_iz = self.compute_iz(virtual_mode=True)
        if self.world.display:
            for polygon_iz in iz:
                if not isinstance(polygon_iz, Polygon) or polygon_iz.exterior is None:
                    continue
                self.world.occupancy_viewer_vis.add_polygon(polygon_iz, color="#0000ff",
                                                            frame=self.world.recorder.current_frame,
                                                            label='TP obstacles'
                                                            )
            for polygon_virtual_iz in virtual_iz:
                if not isinstance(polygon_virtual_iz, Polygon) or polygon_virtual_iz.exterior is None:
                    continue
                self.world.occupancy_viewer_vis.add_polygon(polygon_virtual_iz, color="#ff00ff",
                                                            frame=self.world.recorder.current_frame,
                                                            label='virtual TP \n obstacles'
                                                            )
        speed_plan, s = self.get_speed_from_iz(iz, virtual_iz, path, diagonal_speed)
        if speed_plan is None:
            return None, None
        if isinstance(speed_plan, str):
            return 'in poly', None
        out_path = get_out_path(path, s)
        speed_plan_out = []
        for t, speed in enumerate(speed_plan):
            if s[t][1] > path_length:
                break
            speed_plan_out.append(speed[1])
        speed_plan_out[-1] = 0.0
        return speed_plan_out, out_path

    def compute_iz(self, virtual_mode=False):
        self.world.occupancy_mapper.add_time_gap_to_obstacles(self.time_gap)
        self.world.occupancy_mapper.add_dynamic_signal_to_obstacles()
        intersection, _, rm = self.world.occupancy_mapper.intersection_rs_circles(
            rm_mode=True,
            virtual_mode=virtual_mode,
            eps=self.world.margin ** 2  # eps is the square of the distance
        )
        dynamic_iz_tg = []
        try:
            for poly in get_projected_polygons(intersection, projection_plan='ts'):
                if len(poly) < 3:
                    continue
                else:
                    poly = np.array([point * np.array([self.world.time_step_res, 1]) for point in poly])
                    polygon_iz = Polygon([tuple(point) for point in poly]).convex_hull
                dynamic_iz_tg.append(polygon_iz)
            dynamic_iz_tg = clean_obstacles(dynamic_iz_tg, 0.0, 0.0)
            messy_poly = dynamic_iz_tg
            dynamic_iz = []
            for poly in messy_poly:
                if isinstance(poly, Polygon):
                    if virtual_mode:
                        final_poly = []
                        for point in poly.exterior.coords:
                            final_poly.append(point)
                            final_poly.append([self.world.time_horizon, point[1]])
                        final_poly = Polygon([tuple(point) for point in final_poly]).convex_hull
                    else:
                        final_poly = poly.convex_hull
                    dynamic_iz.append(final_poly)
        except Exception as e:
            print(e)
            raise e
        return dynamic_iz

    def get_speed_from_iz(self, iz, virtual_iz, path, diagonal_speed, eps=None, count=0):
        if eps is None:
            eps = self.safe_time_gap
        collision_type = None
        collision_point = None
        if count > 100:
            return None, None
        iz = clean_obstacles(iz)
        iz = self.filter_iz(iz, eps)
        virtual_iz = clean_obstacles(virtual_iz)
        virtual_iz = self.filter_iz(virtual_iz, eps)
        for poly in iz:
            if isinstance(poly, Polygon) and poly.buffer(2 * eps).contains(Point((0.0, 0.0))):
                return 'in_poly', None
        for poly in virtual_iz:
            if isinstance(poly, Polygon) and poly.buffer(2 * eps).contains(Point((0.0, 0.0))):
                return None, None
        graph, critical_polygons, is_virtual_poly, time_line = self.get_graph(iz, virtual_iz, diagonal_speed, eps)
        prev_critical_polygons = critical_polygons
        if graph is not None:
            smooth_speed_plan, speed_plan = self.smooth_speed_plan(graph)  # sp = [time, speed]
            if speed_plan is None:
                return None, None
            s = get_s_from_speed_plan(smooth_speed_plan)  # s = [time, curvilinear]
            critical_polygons, is_virtual_poly, collision_point, collision_type = check_collision(
                s, iz, virtual_iz, time_line
            )
            if critical_polygons is None and smooth_speed_plan is not None and s is not None:
                # cross cross_security
                if self.world.cross_security:
                    out_path = get_out_path(path, s)
                    junction = None
                    last_point_in_junction = False
                    points_in_junction = []
                    last_point = out_path[-1]
                    for poly in self.world.occupancy_mapper.static_conditional_virtual_obstacles[str(
                            self.world.recorder.current_frame)]:
                        if poly.contains(Point((last_point[0], last_point[1]))):
                            junction = poly
                            points_in_junction.append(last_point)
                            last_point_in_junction = True
                    if last_point_in_junction:
                        for i in range(len(out_path) - 2):
                            last_point = out_path[-(i + 2)]
                            if junction.buffer(eps).contains(Point((last_point[0], last_point[1]))):
                                points_in_junction.append(last_point)
                            else:
                                break
                        previous_point = None
                        distance = 0
                        for point in points_in_junction:
                            if previous_point is None:
                                previous_point = point
                            else:
                                diff = np.add(np.array(point), - np.array(previous_point))
                                distance += np.linalg.norm(diff)
                                previous_point = point
                        self.world.visual_horizon -= distance + eps
                        return self.get_speed_from_iz(
                            iz, virtual_iz, path, diagonal_speed, eps=eps,
                            count=count + 1)
                # ________
                if self.world.display:
                    self.world.occupancy_viewer_vis.add_line(np.array(graph), color='#ff0000',
                                                             frame=self.world.recorder.current_frame,
                                                             label='graph'
                                                             )
                    self.world.occupancy_viewer_vis.add_line(np.array(s), frame=self.world.recorder.current_frame,
                                                             label='s'
                                                             )
                return smooth_speed_plan, s
        if is_virtual_poly:
            lower_s = float('inf')
            for poly in critical_polygons:
                if not isinstance(poly, Polygon):
                    continue
                for point in poly.exterior.coords:
                    if point[1] < lower_s:
                        lower_s = point[1]
            self.world.visual_horizon = lower_s - eps
        else:
            if graph is not None:
                if collision_type is 'under':
                    idx = min(find_nearest_vector(graph, collision_point))
                    if idx < len(graph) - 1:
                        if collision_point[0] > graph[idx][0]:
                            graph0, graph1 = (graph[idx], graph[idx + 1])
                        else:
                            graph0, graph1 = (graph[idx - 1], graph[idx])
                    else:
                        graph0, graph1 = (graph[-2], graph[-1])
                    graph_vector = np.add(graph1, - graph0)
                    point_vector = np.add(collision_point, - graph0)
                    if point_vector[0] == 0 or graph_vector[0] == 0 or \
                            point_vector[1] / point_vector[0] > graph_vector[1] / graph_vector[0]:
                        if not self.emergency_mode:
                            self.emergency_mode = True
                            return self.get_speed_from_iz(
                                iz, virtual_iz, path, diagonal_speed, eps=eps,
                                count=count + 1)
                        else:
                            return None, None
                if collision_type is 'over':
                    idx = min(find_nearest_vector(graph, collision_point))
                    if idx < len(graph) - 1:
                        if collision_point[0] > graph[idx][0]:
                            graph0, graph1 = (graph[idx], graph[idx + 1])
                        else:
                            graph0, graph1 = (graph[idx - 1], graph[idx])
                    else:
                        graph0, graph1 = (graph[-2], graph[-1])
                    graph_vector = graph1 - graph0
                    point_vector = collision_point - graph0
                    if point_vector[1] / point_vector[0] < graph_vector[1] / graph_vector[0] and \
                            prev_critical_polygons is not None:
                        critical_polygons = prev_critical_polygons
            critical_poly_extruded = []
            for poly in critical_polygons:
                for point in poly.exterior.coords:
                    critical_poly_extruded.append(point)
                    point_transform = [-eps, point[1]]
                    critical_poly_extruded.append(point_transform)
                polygon = Polygon([tuple(point) for point in critical_poly_extruded])
                iz.append(polygon.convex_hull)
                if self.world.display:
                    if not isinstance(polygon, Polygon) or polygon.exterior is None:
                        continue
                    self.world.occupancy_viewer_vis.add_polygon(polygon.convex_hull, color="#ff0000",
                                                                frame=self.world.recorder.current_frame,
                                                                label='critical polygons'
                                                                )
        return self.get_speed_from_iz(
            iz, virtual_iz, path, diagonal_speed, eps=eps, count=count + 1)

    def get_max_speed_line(self, t, s, diagonal_speed):
        if (self.world.visual_horizon - s) / diagonal_speed >= (
                self.world.time_horizon - t):
            out_type = 'time'
            out_point = np.array(
                [self.world.time_horizon, s + (self.world.time_horizon - t) * (
                    diagonal_speed)]
            )
        else:
            out_type = 'visual'
            out_point = np.array(
                [t + (self.world.visual_horizon - s) / diagonal_speed,
                 self.world.visual_horizon]
            )
        return LineString([(t, s), (out_point[0], out_point[1])]), out_point, out_type

    def get_graph(self, iz, virtual_iz, diagonal_speed, eps=None):
        if eps is None:
            eps = self.safe_time_gap
        last_point = np.array([0, 0])
        graph = np.array([last_point])
        time_line = np.array([[0, 'under', None]])
        for _ in range(10):
            speed_line, out_point, out_type = self.get_max_speed_line(last_point[0], last_point[1], diagonal_speed)
            if self.world.time_horizon <= last_point[0] + eps or self.world.visual_horizon <= last_point[1] + eps:
                graph, time_line = self.get_brake_end_graph(out_type, out_point, last_point, graph, time_line)
                return graph, None, False, time_line
            intersects_poly, is_virtual_poly = get_intersecting_poly(iz, virtual_iz, speed_line)
            if len(intersects_poly) == 0:
                graph, time_line = self.get_brake_end_graph(out_type, out_point, last_point, graph, time_line)
                return graph, None, False, time_line
            else:
                if is_virtual_poly:
                    return None, intersects_poly, is_virtual_poly, time_line
                else:
                    passing_point = get_cross_min_point(intersects_poly, last_point, eps,
                                                        cross_max=self.world.vehicle_speed_limit)
                    if passing_point is None:
                        return None, intersects_poly, False, time_line
                    time_line_temp = [passing_point[0], 'under', intersects_poly]
                    test_line = LineString([(last_point[0], last_point[1]), (passing_point[0], passing_point[1])])
                    intersects_poly, is_virtual_poly = get_intersecting_poly(iz, virtual_iz, test_line)
                    for _ in range(10):
                        if len(intersects_poly) == 0:
                            break
                        if is_virtual_poly:
                            return None, intersects_poly, is_virtual_poly, time_line
                        passing_point = get_cross_max_point(intersects_poly, last_point, eps)
                        time_line_temp = np.array([passing_point[0], 'over', intersects_poly])
                        test_line = LineString([(last_point[0], last_point[1]), (passing_point[0], passing_point[1])])
                        intersects_poly, is_virtual_poly = get_intersecting_poly(iz, virtual_iz, test_line)
                    time_line = np.vstack((time_line, time_line_temp))
                    if passing_point[0] < graph[-1, 0]:
                        return graph, intersects_poly, is_virtual_poly, time_line
                    graph = np.vstack((graph, passing_point))
                    last_point = passing_point

    def smooth_speed_plan(self, graph):
        count = 1
        last_s = 0
        smooth_speed_plan = np.array([0, self.world.vehicle_speed])
        speed_plan = np.array([0, self.world.vehicle_speed])
        last_acc = self.world.vehicle_acc
        last_speed = self.world.vehicle_speed
        for i in range(int(self.world.time_horizon // self.world.time_step_res)):
            current_time = (i + 1) * self.world.time_step_res
            if count + 1 < len(graph):
                if current_time >= graph[count][0]:
                    count += 1
            if graph[count][0] - current_time == 0:
                target_speed = 0 if graph[count][1] - last_s != 0 else self.world.vehicle_speed_limit
            else:
                target_speed = np.clip(
                    (graph[count][1] - last_s) / (graph[count][0] - current_time),
                    0,
                    self.world.vehicle_speed_limit
                )
            target_acc = np.clip(
                (target_speed - last_speed) / self.world.time_step_res,
                -self.max_acc,
                self.max_acc
            )
            target_jerk = np.clip(
                (target_acc - last_acc) / self.world.time_step_res,
                -self.world.max_jerk,
                self.world.max_jerk
            )
            last_acc += target_jerk * self.world.time_step_res
            last_acc = np.clip(
                last_acc,
                -self.max_acc,
                self.max_acc
            )
            last_speed += last_acc * self.world.time_step_res
            last_speed = np.clip(
                last_speed,
                0,
                self.world.vehicle_speed_limit
            )
            last_s += last_speed * self.world.time_step_res
            smooth_speed_plan = np.vstack(
                (smooth_speed_plan, np.array([(i + 1) * self.world.time_step_res, last_speed])))
            speed_plan = np.vstack((speed_plan, np.array([(i + 1) * self.world.time_step_res, target_speed])))
            if last_s >= graph[-1][1]:
                if len(smooth_speed_plan) > 2 and len(speed_plan) > 2:
                    smooth_speed_plan = smooth_speed_plan[:-1]
                    speed_plan = speed_plan[:-1]
                else:
                    return None, None
                break
        if len(smooth_speed_plan) > 0 and len(speed_plan) > 0:
            return smooth_speed_plan, speed_plan
        else:
            return None, None

    def get_dist_to_brake(self, speed):
        #  Assumption: jerk is -jerk_max, acc is 0 at t = 0, target_acc = -max_acc
        # case 1: target_acc isn't reached
        if speed > 0:
            time_to_brake = math.sqrt(speed / (1 / 2 * self.world.max_jerk))
        else:
            time_to_brake = 0
        dist_to_brake = speed * time_to_brake - 1 / 6 * self.world.max_jerk * time_to_brake ** 3
        # check if target_acc is reached
        if self.world.max_jerk * time_to_brake > self.max_acc:
            time_to_max_acc = self.max_acc / self.world.max_jerk
            time_to_brake = time_to_max_acc + (
                    speed - (1 / 2 * self.world.max_jerk * time_to_max_acc ** 2)) / self.max_acc
            dist_to_brake = (
                    speed * time_to_brake + self.max_acc * time_to_max_acc * time_to_brake - 1 / 2 * self.max_acc *
                    time_to_brake ** 2 - 1 / 2 * self.world.max_jerk * time_to_max_acc ** 2 * time_to_brake)
        return dist_to_brake

    def get_brake_end_graph(self, out_type, out_point, last_point, graph, time_line):
        if out_type == 'time':
            graph = np.vstack((graph, out_point))
            return graph, time_line
        else:
            sec_count = 0
            while True:
                sec_count += 1
                if sec_count > 10:
                    raise Exception('infinite loop')
                if len(graph) <= 2:
                    last_speed = self.world.vehicle_speed_limit
                else:
                    last_speed = self.world.vehicle_speed_limit
                dist_to_brake = self.get_dist_to_brake(last_speed)
                if out_point[1] - dist_to_brake > last_point[1]:
                    time = 0
                    if last_speed != 0:
                        time = dist_to_brake / last_speed
                    if out_point[0] - time > 0:
                        stop_point = [out_point[0] - time, out_point[1] - dist_to_brake]
                        graph = np.vstack((graph, stop_point))
                    else:
                        stop_point = np.array([self.world.time_horizon, out_point[1] - dist_to_brake])
                    out_point = np.array([self.world.time_horizon, out_point[1] - dist_to_brake])
                    graph = np.vstack((graph, out_point))
                    horizon_point = np.array([self.world.time_horizon, self.world.visual_horizon])
                    graph = np.vstack((graph, horizon_point))
                    break
                else:
                    if len(graph) > 2:
                        graph = np.delete(graph, -1, 0)
                        time_line = np.delete(time_line, -1, 0)
                        last_point = graph[-1]
                    else:
                        last_point = np.array([0, 0])
                        stop_point = [self.world.time_horizon, 0]
                        graph = np.vstack((last_point, stop_point))
                        horizon_point = [self.world.time_horizon, self.world.visual_horizon]
                        graph = np.vstack((graph, horizon_point))
                        break
        time_line_temp = np.array([stop_point[0], 'under', None])
        time_line = np.vstack((time_line, time_line_temp))
        return graph, time_line

    def filter_iz(self, iz, eps):
        limit_poly = [
            [eps, eps],
            [self.world.time_horizon - eps, eps],
            [self.world.time_horizon - eps, self.world.visual_horizon - eps],
            [eps, self.world.visual_horizon - eps]
        ]
        limit_poly = Polygon(list(point) for point in limit_poly)
        out = []
        for poly in iz:
            if not isinstance(poly, Polygon):
                continue
            out.append(limit_poly.intersection(poly))
        return out
