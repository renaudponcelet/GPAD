import math
import os
import pickle
import threading

# import matplotlib.pyplot as plt
import numpy as np
from PythonAPI.GPAD.Common.Utils.carla_utils import carla_vector2array_2d, get_bounding_box_shape_circles, \
    dist_location_2d, lateral_shift
from PythonAPI.GPAD.Common.Utils.utils import path2rs, dist, clean_obstacles, get_secure_dist
from PythonAPI.GPAD.Common.occupancy_mapper.ReachableSetGenerator import ReachableSetGenerator
from PythonAPI.GPAD.Common.occupancy_mapper.ReachableSetPredictor import get_reachable_waypoints
from shapely.geometry import Point, LineString, Polygon


class TrackerUpdate(threading.Thread):
    def __init__(self, world, tracker, frame):
        super(TrackerUpdate, self).__init__()
        self.tracker = tracker
        self.world = world
        self.frame = frame
        self.obstacles = None

    def run(self):
        self.obstacles = tracker_update(self.tracker, self.world)


def tracker_update(tracker, world):
    tracker.update()
    obstacles = None
    if tracker.visible:
        obstacles = np.full(len(tracker.future_pos_estimation), 0.0, dtype=object)
        for t, time_slice in enumerate(tracker.future_pos_estimation):
            slice_circles = None
            for pos in time_slice:
                yaw = pos[2]
                rot = np.array([
                    [math.cos(yaw), math.sin(yaw), 0],
                    [-math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]
                ]).round(decimals=2)
                d = get_secure_dist(
                    np.deg2rad(world.occupancy_mapper.vehicle_profile["specific"]["max_steer_angle"]),
                    world.global_path_interval,
                    world.occupancy_mapper.vehicle_profile["specific"]["wheel_spacing"],
                    world.vehicle.bounding_box.extent.y
                )
                shape = np.array([tracker.vehicle_shape_circles[0], tracker.vehicle_shape_circles[1],
                                  np.add([- d, 0, 0], tracker.vehicle_shape_circles[1])])
                circles = shape.dot(rot)
                circles = np.add(circles, np.array([pos[0], pos[1], tracker.vehicle_shape_radius]).round(decimals=2))
                if slice_circles is None:
                    slice_circles = circles
                else:
                    slice_circles = np.concatenate((slice_circles, circles), axis=0)
                if world.display:
                    if t == 0:
                        for circle in slice_circles:
                            world.occupancy_viewer_ris.add_circle(circle,
                                                                  frame=world.recorder.current_frame,
                                                                  screens=[0, 1, 2, 3],
                                                                  label='vehicle'
                                                                  )
            obstacles[t] = slice_circles
    return obstacles


class OccupancyMapper:
    name = "class to build occupancy map with semantic segmentation from carla sensors"

    def __init__(self, world):
        self.world = world
        self.static_map = None
        self.get_road_map()
        self.junctions = None
        self.traffic_light_state = None
        self.get_junctions()
        if world.awareness == "vv":
            self.virtual_vehicle_generator = None
        self.virtual_obstacles = {}
        self.static_conditional_virtual_obstacles = {}
        self.rs = None
        self.static_ris = None
        self.rm = None
        self.dynamic_obstacles_margin = {}
        self.dynamic_obstacles_margin_signal = {}
        self.frame_buffer = 100
        if world.awareness == "omniscient":
            self.omni_light = 1
        else:
            self.omni_light = 0
        # be careful if frame_buffer is too low and simulation is too slow the frame can be delete too soon
        self.static_poly = {}
        self.dynamic_obstacles = {}
        self.signal_obstacles = {}
        self.vehicle_profile = {
            "specific": self.world.vehicles_info["specific"][self.world.vehicle.type_id],
            "general": self.world.vehicles_info["general"]
        }
        self.generator = ReachableSetGenerator(
            world,
            u_max=self.vehicle_profile["specific"]["max_steer_angle"],
            wheel_spacing=self.vehicle_profile["specific"]["wheel_spacing"]
        )
        self.tracker_list = None

    def final_obstacles(self):
        if str(self.world.recorder.current_frame) in self.dynamic_obstacles_margin_signal:
            final_obstacles = self.dynamic_obstacles_margin_signal[str(self.world.recorder.current_frame)]
        elif str(self.world.recorder.current_frame) in self.dynamic_obstacles_margin:
            final_obstacles = self.dynamic_obstacles_margin[str(self.world.recorder.current_frame)]
        elif str(self.world.recorder.current_frame) in self.dynamic_obstacles:
            final_obstacles = self.dynamic_obstacles[str(self.world.recorder.current_frame)]
        else:
            raise KeyError
        return final_obstacles

    def set_virtual_vehicle_generator(self, virtual_vehicle_generator):
        self.virtual_vehicle_generator = virtual_vehicle_generator

    def update_obstacles(self):
        if str(self.world.recorder.current_frame) in self.static_poly:
            return
        self.static_poly[str(self.world.recorder.current_frame)] = self.static_map
        self.static_conditional_virtual_obstacles[str(self.world.recorder.current_frame)] = self.junctions
        self.signal_obstacles[str(self.world.recorder.current_frame)], self.traffic_light_state = (
            self.update_signal_obstacles())
        self.dynamic_obstacles[str(self.world.recorder.current_frame)] = self.update_dynamic_obstacles()
        if self.world.awareness == "blind" or self.world.awareness == 'omniscient':
            self.virtual_obstacles[str(self.world.recorder.current_frame)] = np.empty(0)
        else:
            self.virtual_obstacles[str(self.world.recorder.current_frame)] = self.update_virtual_obstacles()
        if self.world.display:
            for poly in self.static_map:
                self.world.occupancy_viewer_ris.add_polygon(poly, color='#a0a0a0',
                                                            frame=self.world.recorder.current_frame,
                                                            screens=[0, 1, 2, 3],
                                                            label='static map'
                                                            )

    def update_rs(self, static_path, width, speed_plan, speed_profile=None, blocked=False):
        self.rs, self.static_ris = self.generator.run(static_path, width, speed_plan,
                                                      speed_profile=speed_profile, blocked=blocked)

    def update_rm(self, path, pos, vis_speed):
        rm_out = []
        path_length = 0
        precedent_point = None
        for point in path:
            if precedent_point is None:
                precedent_point = point
            else:
                path_length += dist(point, precedent_point)
                precedent_point = point
        dist_to_brake = vis_speed.get_dist_to_brake(self.world.vehicle_speed_limit)
        if path_length > dist_to_brake:
            diagonal_speed = self.world.vehicle_speed_limit
        else:
            t_switch = 1 / 2 * (self.world.time_horizon - self.world.vehicle_speed / self.world.max_acc)
            diagonal_speed = self.world.vehicle_speed + self.world.max_acc * t_switch
        if diagonal_speed > self.world.vehicle_speed_limit:
            diagonal_speed = self.world.vehicle_speed_limit
        speed_plan = np.full(int(self.world.time_horizon // self.world.time_step_res), diagonal_speed)
        yaw = np.deg2rad(self.world.vehicle.get_transform().rotation.yaw)
        path_rs = path2rs(np.array(path), carla_vector2array_2d(pos).round(decimals=2), yaw, speed_plan,
                          self.world.time_step_res,
                          self.world.global_path_interval,
                          np.deg2rad(self.vehicle_profile["specific"]["max_steer_angle"]),
                          self.vehicle_profile["specific"]["wheel_spacing"], self.world.scenario_data["L"],
                          self.world.visual_horizon,
                          nb_circles=self.world.nb_circles, ego_offset=self.world.ego_offset, sort=True)
        rs_0 = path_rs[0][0]
        if self.world.display:
            for circle in rs_0:
                self.world.occupancy_viewer_ris.add_circle([circle[1], circle[2], self.world.ego_radius],
                                                           color='#550000',
                                                           frame=self.world.recorder.current_frame,
                                                           screens=[0, 1, 2, 3],
                                                           label='ego vehicle'
                                                           )
        path_rs = np.array(path_rs)
        shape = path_rs.shape
        path_rs = np.concatenate(path_rs, axis=0)
        for i in range(shape[0]):
            rm_out.append(path_rs)
        self.rm = np.array(rm_out)
        # if len(self.rm) < int(self.world.time_horizon // self.world.time_step_res):
        #     print("Warning : rm length too low")
        return diagonal_speed, path_length

    def set_trackers(self, tracker_list):
        self.tracker_list = tracker_list

    def update_signal_obstacles(self):
        traffic_light = None
        ego_waypoint = self.world.map.get_waypoint(self.world.vehicle.get_location())
        if self.world.planner.local_lane[0]['status'] is 'o':
            if ego_waypoint.is_junction:
                ego_location_global_hash = ego_waypoint.junction_id
            else:
                ego_location_global_hash = hash((str(ego_waypoint.road_id),
                                                 str(ego_waypoint.section_id)))
            if ego_location_global_hash in self.world.planner.global_paths_alternative:
                for hash_tag in (
                        self.world.planner.global_paths_alternative[ego_location_global_hash]['alternative_lane']):
                    if (
                            self.world.planner.global_paths_alternative[ego_location_global_hash]['alternative_lane'][
                                hash_tag]['status'] is 's'
                    ):
                        ego_waypoint = (
                            self.world.planner.global_paths_alternative[ego_location_global_hash]['alternative_lane'][
                                hash_tag]['lane'][0]
                        )
        landmarks = ego_waypoint.get_landmarks_of_type(self.world.visual_horizon, '1000001', stop_at_junction=False)
        min_dist = float('inf')
        for landmark in landmarks:
            landmark_waypoint = landmark.waypoint
            if landmark_waypoint.is_junction:
                global_hash = landmark_waypoint.junction_id
            else:
                global_hash = hash((str(landmark_waypoint.road_id), str(landmark_waypoint.section_id)))
            landmark_hash = hash(
                (str(landmark_waypoint.road_id), str(landmark_waypoint.section_id), str(landmark_waypoint.lane_id))
            )
            if global_hash in self.world.planner.global_paths_alternative \
                    and landmark_hash in self.world.planner.global_paths_alternative[global_hash]['alternative_lane']:
                distance = dist_location_2d(self.world.vehicle.get_location(), landmark_waypoint.transform.location)
                if distance < min_dist:
                    traffic_light = self.world.world.get_traffic_light(landmark)
                    min_dist = distance
        if traffic_light is not None:
            pos1 = carla_vector2array_2d(traffic_light.bounding_box.location).round(decimals=2)
            yaw1 = np.deg2rad(traffic_light.bounding_box.rotation.yaw)
            rot1 = np.array([
                [math.cos(yaw1), math.sin(yaw1), 0],
                [-math.sin(yaw1), math.cos(yaw1), 0],
                [0, 0, 1]
            ]).round(decimals=2)
            shape, radius = get_bounding_box_shape_circles(
                12 * traffic_light.bounding_box.extent.x, traffic_light.bounding_box.extent.y, 4)
            circles_bb = shape.dot(rot1)
            circles_bb = np.add(
                circles_bb, np.array([pos1[0], pos1[1], radius]).round(decimals=2))
            pos2 = carla_vector2array_2d(traffic_light.get_location()).round(decimals=2)
            yaw2 = np.deg2rad(traffic_light.get_transform().rotation.yaw)
            rot2 = np.array([
                [math.cos(yaw2), math.sin(yaw2), 0],
                [-math.sin(yaw2), math.cos(yaw2), 0],
                [0, 0, 1]
            ]).round(decimals=2)
            circles = circles_bb.dot(rot2)
            circles = np.add(
                circles, np.array([pos2[0], pos2[1], radius]).round(decimals=2))
            if str(traffic_light.state) == 'Red' or str(traffic_light.state) == 'Yellow':
                if self.world.display:
                    color = '#ff0000' if str(traffic_light.state) == 'Red' else '#ffff00'
                    label = 'traffic light at red' if str(traffic_light.state) == 'Red' else 'traffic light at yellow'
                    for circle in circles:
                        self.world.occupancy_viewer_ris.add_circle(circle, color, self.world.recorder.current_frame,
                                                                   screens=[0, 1, 2, 3], label=label)
                if self.world.planner.local_lane[0]['status'] is 'o':
                    time_to_red = 0 if str(traffic_light.state) == 'Red' else \
                        (traffic_light.get_yellow_time() - self.omni_light * traffic_light.get_elapsed_time())
                else:
                    time_to_red = 0 if str(traffic_light.state) == 'Red' else \
                        (traffic_light.get_yellow_time() / 2 - self.omni_light * traffic_light.get_elapsed_time())
                if time_to_red < 0:
                    time_to_red = 0
                if str(traffic_light.state) == 'Yellow':
                    time_to_green = traffic_light.get_red_time() + traffic_light.get_yellow_time()
                else:
                    time_to_green = traffic_light.get_red_time()
                time_to_green -= self.omni_light * traffic_light.get_elapsed_time()
            elif str(traffic_light.state) == 'Green':
                if self.world.display:
                    color = '#00ff00'
                    for circle in circles:
                        self.world.occupancy_viewer_ris.add_circle(circle,
                                                                   color,
                                                                   self.world.recorder.current_frame,
                                                                   screens=[0, 1, 2, 3],
                                                                   label='traffic light \n at green'
                                                                   )
                time_to_red = (traffic_light.get_green_time() + traffic_light.get_yellow_time() / 2
                               - self.omni_light * traffic_light.get_elapsed_time())
                time_to_green = (traffic_light.get_green_time() + traffic_light.get_yellow_time()
                                 + traffic_light.get_red_time() - self.omni_light * traffic_light.get_elapsed_time())
            else:
                print("traffic_light is in a undetermined state !")
                time_to_red = -1
                time_to_green = 1000
                circles = []
            step_to_red = time_to_red / self.world.time_step_res
            step_to_green = time_to_green / self.world.time_step_res
            obstacle = []
            for t in range(self.world.nb_step):
                if t < step_to_red:
                    obstacle.append(None)
                elif t < step_to_green:
                    obstacle.append(circles)
                else:
                    obstacle.append(None)
        else:
            obstacle = []
        if traffic_light is not None:
            return np.array(obstacle), traffic_light.state
        else:
            return np.array(obstacle), None

    def update_dynamic_obstacles(self):
        thread_list = []
        obstacles = None
        if self.tracker_list is None:
            return None
        if self.world.allow_threading:
            for tracker in self.tracker_list:
                thread = TrackerUpdate(
                    self.world,
                    tracker,
                    self.world.recorder.current_frame
                )
                thread.start()
                thread_list.append(thread)
            for thread in thread_list:
                thread.join(timeout=6)
                if thread.obstacles is not None:
                    if obstacles is None:
                        obstacles = thread.obstacles
                    else:
                        for t, slide in enumerate(thread.obstacles):
                            obstacles[t] = np.concatenate((obstacles[t], slide), axis=0)
        else:
            for tracker in self.tracker_list:
                tracker_obstacles = tracker_update(tracker, self.world)
                if tracker_obstacles is not None:
                    if obstacles is None:
                        obstacles = tracker_obstacles
                    else:
                        for t, slide in enumerate(tracker_obstacles):
                            obstacles[t] = np.concatenate((obstacles[t], slide), axis=0)
        return obstacles

    def update_virtual_obstacles(self):
        if self.virtual_vehicle_generator is None:
            return []
        dangerous_waypoints = self.virtual_vehicle_generator.get_dangerous_waypoints()
        critical_waypoints = self.virtual_vehicle_generator.get_critical_waypoints(dangerous_waypoints)
        virtual_obstacles = None
        for waypoint in critical_waypoints:
            pos = carla_vector2array_2d(waypoint.transform.location).round(decimals=2)
            yaw = np.deg2rad(waypoint.transform.rotation.yaw)
            direction = np.array([math.cos(yaw), math.sin(yaw)]).round(decimals=2)
            speed = self.world.vehicle_speed_limit
            for tracker in self.tracker_list:
                if dist(tracker.pos, pos) < self.world.vehicle_speed_limit * self.world.time_horizon:
                    dir_vector = np.add(tracker.pos[:1], -pos)
                    track_yaw = np.arctan2(dir_vector[1], dir_vector[0])
                    if track_yaw > 2 * math.pi:
                        track_yaw -= 2 * math.pi
                    if track_yaw <= 0:
                        track_yaw += 2 * math.pi
                    if (track_yaw - yaw) % (2 * math.pi) < math.radians(10):
                        speed = tracker.mean_speed
            if speed is None:
                speed = self.world.vehicle_speed_limit
            spawn_point = pos - waypoint.lane_width / 2 * direction
            # spawn_point = pos
            reachable_waypoints, _ = get_reachable_waypoints(
                self.world,
                spawn_point,
                speed,
                virtual_mode=True
            )
            if self.world.display:
                self.world.occupancy_viewer_ris.add_square([spawn_point[0], spawn_point[1]], color='#550055',
                                                           frame=self.world.recorder.current_frame,
                                                           screens=[0, 1, 2, 3],
                                                           label='virtual vehicle'
                                                           )
            obstacles = np.full(len(reachable_waypoints), 0.0, dtype=object)
            count = 0
            for t, time_slice in enumerate(reachable_waypoints):
                slice_circles = None
                if self.world.display:
                    if count == 0:
                        for pos in time_slice:
                            self.world.occupancy_viewer_ris.add_point(np.array([pos[0], pos[1]]).round(decimals=2),
                                                                      color='#ff00ff',
                                                                      frame=self.world.recorder.current_frame,
                                                                      screens=[0, 1, 2, 3],
                                                                      label='virtual reachable \n set'
                                                                      )
                        count += 1
                    elif count == 5:
                        count = 0
                    else:
                        count += 1
                for pos in time_slice:
                    yaw = pos[2]
                    rot = np.array([
                        [math.cos(yaw), math.sin(yaw), 0],
                        [-math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1]
                    ]).round(decimals=2)
                    virtual_vehicle_shape_circles, virtual_vehicle_shape_radius = get_bounding_box_shape_circles(
                        waypoint.lane_width,
                        waypoint.lane_width / 2
                    )
                    circles = virtual_vehicle_shape_circles.dot(rot)
                    circles = np.add(
                        circles, np.array([pos[0], pos[1], virtual_vehicle_shape_radius]).round(decimals=2))
                    if slice_circles is None:
                        slice_circles = circles
                    else:
                        slice_circles = np.concatenate((slice_circles, circles), axis=0)
                obstacles[t] = slice_circles
            if virtual_obstacles is None:
                virtual_obstacles = obstacles
            else:
                for t, slide in enumerate(obstacles):
                    virtual_obstacles[t] = np.concatenate((virtual_obstacles[t], slide), axis=0)
        if virtual_obstacles is None:
            virtual_obstacles = []
        return virtual_obstacles

    def add_time_gap_to_obstacles(self, time_gap):
        if self.dynamic_obstacles[str(self.world.recorder.current_frame)] is None:
            self.dynamic_obstacles_margin[str(self.world.recorder.current_frame)] = None
        else:
            obstacles_out = []
            min_time_gap = time_gap
            for tracker in self.tracker_list:
                if tracker.time_gap < min_time_gap:
                    min_time_gap = tracker.time_gap
            print(min_time_gap)
            for t, obstacles_slice in enumerate(self.dynamic_obstacles[str(self.world.recorder.current_frame)]):
                offset = int(min_time_gap // self.world.time_step_res)
                obstacles_slice_new = []
                for i in range(int(np.clip(t - offset, 0, self.world.time_horizon // self.world.time_step_res - 1)),
                               int(np.clip(t + offset, 0, self.world.time_horizon // self.world.time_step_res - 1))):
                    for circle in self.dynamic_obstacles[str(self.world.recorder.current_frame)][i]:
                        obstacles_slice_new.append(circle)
                obstacles_slice_new = self.simplify_occupation(np.array(obstacles_slice_new))
                obstacles_out.append(np.array(obstacles_slice_new))
            self.dynamic_obstacles_margin[str(self.world.recorder.current_frame)] = np.array(obstacles_out)

    def add_dynamic_signal_to_obstacles(self):
        if self.signal_obstacles[str(self.world.recorder.current_frame)] is None:
            return
        obstacles_out = np.copy(self.dynamic_obstacles_margin[str(self.world.recorder.current_frame)])
        for t, signal_slice in enumerate(self.signal_obstacles[str(self.world.recorder.current_frame)]):
            if signal_slice is not None and len(signal_slice) > 0:
                obstacles_out[t] = np.concatenate(
                    (self.dynamic_obstacles_margin[str(self.world.recorder.current_frame)][t],
                     signal_slice),
                    axis=0)
        self.dynamic_obstacles_margin_signal[str(self.world.recorder.current_frame)] = obstacles_out

    def intersection_rs_circles(self, rs=None, eps=0.1, rm_mode=False, virtual_mode=False):
        # can be used with rm (risk matrix) for speed planning
        try:
            if not virtual_mode:
                obstacles = self.final_obstacles()
            else:
                obstacles = self.virtual_obstacles[str(self.world.recorder.current_frame)]
        except KeyError:
            obstacles = None
        if rs is None:
            if not rm_mode:
                if self.rs is None:
                    return [], [], []
                rs = self.rs
            else:
                if self.rm is None:
                    return [], [], []
                rs = self.rm
        else:
            rs = np.array(rs)
        if obstacles is None or len(obstacles.shape) == 0 or obstacles.shape[0] == 0 or \
                self.world.ego_radius is None or rs.size == 0:
            rs_vertices = []
            for t in range(len(rs)):
                rs_vertices.append([])
                rs_vertices[t].append([])
                for i in range(len(rs[t])):
                    rs_vertices[t][0].append(rs[t][i][0])
            return [], rs_vertices, rs
        eps += self.world.ego_radius
        copy = np.copy(rs)
        out = np.full(copy.shape[0], 0., dtype=object)
        a_out = np.full(copy.shape[0], 0., dtype=object)
        for i, obj in enumerate(copy):
            out[i] = []
            obj = np.array(obj)
            if i >= obstacles.shape[0]:
                continue
            if obj.size == 0:
                rs_vertices = []
                for t in range(len(rs)):
                    rs_vertices.append([])
                    rs_vertices[t].append([])
                    for j in range(len(rs[t])):
                        rs_vertices[t][0].append(rs[t][j][0])
                return [], rs_vertices, rs
            if isinstance(obstacles[i], float):
                continue
            # get x = [x, y] from [0, x, y, 0] from each circle
            x = np.concatenate((obj[:, 1:, :].dot(np.array([[0, 0], [1, 0], [0, 1], [0, 0]]))), axis=0)
            c = obstacles[i].dot(np.array([[1, 0], [0, 1], [0, 0]]))
            x_x = np.transpose(x)[0].reshape((x.shape[0], 1))
            x_y = np.transpose(x)[1].reshape((x.shape[0], 1))
            if c.shape != (2,):
                c_x = np.transpose(c)[0].reshape((c.shape[0], 1))
                c_y = np.transpose(c)[1].reshape((c.shape[0], 1))
            else:
                c_x = c[0]
                c_y = c[1]
            diff_x = np.add(x_x, - c_x.T)
            diff_y = np.add(x_y, - c_y.T)
            dist2 = diff_x ** 2 + diff_y ** 2
            r = obstacles[i].dot(np.array([0, 0, 1]))
            val = np.sqrt(dist2) - r
            test = val <= eps
            # test = np.logical_or(test[0::2], test[not 0::2])
            test = np.logical_or.reduce([test[i::self.world.nb_circles] for i in range(self.world.nb_circles)])
            # test = test.any(axis=0)
            flag = None
            i_temp = []
            a_i_temp = []
            j_temp = []
            a_j_temp = []
            for k, v in enumerate(test):
                obj_k = obj[k][0]
                if flag is None:
                    if v.any():
                        flag = True
                        j_temp = [obj_k]
                    else:
                        flag = False
                        a_j_temp = [obj_k]
                else:
                    if v.any() and not flag:
                        flag = True
                        a_i_temp.append(a_j_temp)
                        j_temp = [obj_k]
                    elif v.any() and flag:
                        j_temp.append(obj_k)
                    elif not v.any() and flag:
                        flag = False
                        i_temp.append(np.array(j_temp))
                        a_j_temp = [obj_k]
                    elif not v.any() and not flag:
                        a_j_temp.append(obj_k)
                    else:
                        raise Exception("Maybe an value error")
            if flag:
                i_temp.append(np.array(j_temp))
            else:
                a_i_temp.append(np.array(a_j_temp))
            out[i] = np.array(i_temp)
            a_out[i] = np.array(a_i_temp)
        return out, a_out, rs

    def in_map(self, pos_array, eps=0, screen=None):
        if screen is None:
            screen = [0, 1, 2, 3]
        for pos in pos_array:
            for poly in self.static_poly[str(self.world.recorder.current_frame)]:
                if Point((pos[0], pos[1])).buffer(eps).intersects(poly):
                    if self.world.display:
                        self.world.occupancy_viewer_ris.add_point([pos[0], pos[1]],
                                                                  screens=screen,
                                                                  frame=self.world.recorder.current_frame,
                                                                  label='rs in map'
                                                                  )
                    return True
        return False

    def clean_old_frame(self):
        frame2del = []
        for frame_index in self.static_poly:
            if self.world.recorder.current_frame - int(frame_index) >= self.frame_buffer:
                frame2del.append(frame_index)
        for frame in frame2del:
            del self.static_poly[frame]
            del self.dynamic_obstacles[frame]

    def get_road_map(self, recompute=False):
        land_path = None
        if self.world.scenario_mode:
            if os.path.isfile(self.world.scenario_path + self.world.scenario_name + '/land'):
                land_path = self.world.scenario_path + self.world.scenario_name + '/land'
            elif '/' in self.world.scenario_name:
                super_scenario_name = self.world.scenario_name[:self.world.scenario_name.find('/')]
                if os.path.isfile(self.world.scenario_path + super_scenario_name + '/land'):
                    land_path = self.world.scenario_path + super_scenario_name + '/land'

        if land_path is not None and not recompute:
            # Load polygon from disc
            with open(land_path, "rb") as poly_file:
                land = pickle.load(poly_file)
        else:
            road_list = []
            margin = 50
            max_x = max(self.world.waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
            max_y = max(self.world.waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
            min_x = min(self.world.waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
            min_y = min(self.world.waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin
            for hash_tag in self.world.waypoints_dic:
                waypoints = self.world.waypoints_dic[hash_tag]
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]
                polygon = road_left_side + [x for x in reversed(road_right_side)]
                road_list.append(Polygon([(point.x, point.y) for point in polygon]))
            road = clean_obstacles(road_list, 6 * self.world.margin,
                                   self.world.margin)
            land = Point(((max_x + min_x) / 2, (max_y + min_y) / 2)).buffer(
                max((max_x - min_x), (max_y - min_y)),
                resolution=1
            )
            for poly in road:
                land = land.difference(poly)
            land = clean_obstacles(
                [land],
                0,
                0
            )
            # if self.world.display:
            #     for poly in land:
            #         x, y = poly.exterior.xy
            #         plt.plot(x, y)
            #         for interior in poly.interiors:
            #             x, y = interior.xy
            #             plt.plot(x, y)
            #     plt.show()
            # Save polygon to disc
            # land is always store in scenario directory you can move the land file to add it to the super_scenario
            if self.world.scenario_mode:
                with open(self.world.scenario_path + self.world.scenario_name + '/land', "wb") as poly_file:
                    pickle.dump(land, poly_file, pickle.HIGHEST_PROTOCOL)
        sorted_land = [None]
        for poly in land:
            if len(poly.interiors) > 0:
                sorted_land[0] = poly
            else:
                sorted_land.append(poly)
        if sorted_land[0] is None:
            del sorted_land[0]
        self.static_map = sorted_land

    def get_junctions(self, recompute=False):
        junction_path = None
        if self.world.scenario_mode:
            if os.path.isfile(self.world.scenario_path + self.world.scenario_name + '/junction'):
                junction_path = self.world.scenario_path + self.world.scenario_name + '/junction'
            elif '/' in self.world.scenario_name:
                super_scenario_name = self.world.scenario_name[:self.world.scenario_name.find('/')]
                if os.path.isfile(self.world.scenario_path + super_scenario_name + '/junction'):
                    junction_path = self.world.scenario_path + super_scenario_name + '/junction'
        if junction_path is not None and not recompute:
            # Load polygon from disc
            with open(junction_path, "rb") as poly_file:
                self.junctions = pickle.load(poly_file)
        else:
            waypoint_list_point = []
            road = None
            for waypoint in self.world.intersection_waypoints:
                waypoint_list_point.append(carla_vector2array_2d(waypoint.transform.location).round(decimals=2))
                if road is None:
                    road = Point(list(waypoint_list_point[-1])).buffer(waypoint.lane_width / 2)
                else:
                    road = road.union(Point(list(waypoint_list_point[-1])).buffer(waypoint.lane_width / 2))
            junction_list = []
            for poly in road:
                junction_list.append(poly.buffer(self.world.margin).convex_hull)
            if self.world.scenario_mode:
                # junction is always store in scenario directory you can move the junction file to add it to the
                # super_scenario
                with open(self.world.scenario_path + self.world.scenario_name + '/junction', "wb") as poly_file:
                    pickle.dump(junction_list, poly_file, pickle.HIGHEST_PROTOCOL)
            self.junctions = junction_list

    def simplify_occupation(self, circle_collection):
        if len(circle_collection) < 3:
            return circle_collection
        circle_collection = np.unique(circle_collection.round(decimals=2), axis=0)
        # We suppose all radius are equals
        if len(circle_collection) < 3:
            return circle_collection
        r = circle_collection[0][2]
        poly = Polygon([list(point[:2]) for point in circle_collection])
        poly_ch = poly.convex_hull
        if isinstance(poly_ch, LineString):
            simplified_occupation = []
            for point in np.array(poly_ch.coords):
                simplified_occupation.append([point[0], point[1], r])
            return np.unique(simplified_occupation, axis=0)
        else:
            poly_s = poly.simplify(self.world.margin).exterior
            simplified_occupation = []
            for point in np.array(poly_s.coords):
                simplified_occupation.append([point[0], point[1], r])
            return np.unique(simplified_occupation, axis=0)
