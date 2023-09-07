import carla
import numpy as np
from GPAD.Approaches.Common.visibility_lazy import clean_obstacles
from GPAD.Common.Utils.carla_utils import carla_vector2array_2d, get_cost, Set, Graph
from GPAD.Common.Utils.utils import find_paths, path2rs, SpeedPlan, get_projected_polygons, get_realistic_path
from GPAD.Common.Utils.agents.tools.misc import get_speed
from shapely.geometry import Polygon


class MMRIS:
    name = "MMRIS class for Carla simulator"

    def __init__(self, world):
        self.world = world
        self.path = None
        self.__actual_speed = None
        self.planner = None
        self.speed_plan = []
        self.speed_profile = None
        self.screen = 0

    def get_path(self, planner, static_path, speed_profile, goal, recompute=False):
        self.planner = planner
        self.speed_profile = speed_profile
        if self.world.display:
            if speed_profile == 'brake':
                self.screen = 1
            elif speed_profile == 'keep_speed':
                self.screen = 2
            else:
                self.screen = 3
        if not recompute:
            max_acceleration = self.world.max_acc
            deceleration = max_acceleration / 2
            self.speed_plan = self.get_speed_from_profile(
                self.actual_speed,
                speed_profile,
                max_acceleration,
                self.world.vehicle_speed_limit,
                deceleration
            )
            if self.world.display:
                self.world.occupancy_viewer_ris.add_line(static_path,
                                                         color='#00ffff',
                                                         frame=self.world.recorder.current_frame,
                                                         screens=self.screen,
                                                         label='static path'
                                                         )
            static_path = path2rs(static_path[:, :2], static_path[0, :2], static_path[0, 2], self.speed_plan,
                                  self.world.time_step_res,
                                  self.world.global_path_interval,
                                  np.deg2rad(
                                      self.world.occupancy_mapper.vehicle_profile["specific"]["max_steer_angle"]),
                                  self.world.occupancy_mapper.vehicle_profile["specific"]["wheel_spacing"],
                                  self.world.scenario_data["L"],
                                  self.world.visual_horizon,
                                  index_limit=self.world.nb_step, flat=True)
            # if self.world.display:
            #     for point in static_path:
            #         self.world.occupancy_viewer_ris.add_circle(
            #         [point[0], point[1], 1], color='#000000', frame=frame, screens=self.screen)
            self.path = static_path[:, :2]
        width = 6 * self.world.vehicle.bounding_box.extent.y
        self.world.occupancy_mapper.update_rs(
            static_path, width, self.speed_plan, speed_profile, blocked=planner.blocked
        )
        try:
            ris, rfs = self.compute_rfs()
            if len(rfs) == 0:
                raise Exception("No rfs")
            path = self.get_path_from_rfs(rfs)
        except Exception as e:
            print(e)
            path = None
            rfs = None
            ris = None
        self.path = path
        if self.path is None and rfs is not None and ris is not None:
            if ris is None:
                self.path = None
            else:
                self.path = self.get_path_from_ris_wrapper(ris, rfs, goal, self.screen)
        if isinstance(self.path, str):
            self.path = None
        return self.path

    @property
    def actual_speed(self):
        self.actual_speed = get_speed(self.world.vehicle) / 3.6
        return self.__actual_speed

    @actual_speed.setter
    def actual_speed(self, speed):
        self.__actual_speed = speed

    def get_path_from_rfs(self, rfs):
        nb_arrival_group = len(rfs[-1])
        if len(rfs[-1]) == 0 or (nb_arrival_group == 1 and len(rfs[-1][0]) == 0):
            nb_arrival_group = 0
        if nb_arrival_group == 0:
            return "no way"

        rfs_graph = self.get_graph_from_rfs(rfs)
        if rfs_graph is None:
            return "no way"
        path = self.get_path_from_graph(rfs_graph, nb_arrival_group)
        return path

    def get_graph_from_rfs(self, rfs):
        # each point of the graph is indexed by timestamp and group index
        graph = {'00': {'children': [], 'rfs_indexes': np.arange(2 * self.world.nb_paths + 1),
                        'group': [0, self.world.vehicle.get_location().x, self.world.vehicle.get_location().y,
                                  np.deg2rad(self.world.vehicle.get_transform().rotation.yaw)],
                        'width': 2 * self.world.nb_paths, 'free_path': np.arange(2 * self.world.nb_paths + 1)}}
        for t in range(int(self.world.time_horizon // self.world.time_step_res)):
            if t == 0:
                continue
            if len(rfs[t]) == 0:
                return None
            for i, group_i in enumerate(rfs[t]):
                if len(group_i) == 0:
                    graph[str(t) + str(i)] = {'children': [], 'rfs_indexes': [], 'group': np.array([]),
                                              'width': 0}
                else:
                    rfs_indexes = np.array(group_i)[:, 0]
                    graph[str(t) + str(i)] = {'children': [], 'rfs_indexes': rfs_indexes, 'group': np.array(rfs[t][i]),
                                              'width': rfs_indexes[-1] - rfs_indexes[0]}
                group_width = []
                for j, group_j in enumerate(rfs[t - 1]):
                    links_width = np.isin(
                        graph[str(t - 1) + str(j)]['rfs_indexes'], graph[str(t) + str(i)]['rfs_indexes']
                    )
                    if 'free_path' in graph[str(t - 1) + str(j)]:
                        links = np.isin(
                            graph[str(t - 1) + str(j)]['free_path'], graph[str(t) + str(i)]['rfs_indexes']
                        )
                        if links.any():
                            if 'free_path' in graph[str(t) + str(i)]:
                                graph[str(t) + str(i)]['free_path'] = np.concatenate(
                                    (graph[str(t) + str(i)]['free_path'],
                                     graph[str(t - 1) + str(j)]['free_path'][links]),
                                    axis=0
                                )
                            else:
                                graph[str(t) + str(i)]['free_path'] = graph[str(t - 1) + str(j)]['free_path'][links]
                    if links_width.any():
                        group_width.append(graph[str(t - 1) + str(j)]['width'])
                        graph[str(t - 1) + str(j)]['children'].append(str(t) + str(i))
                if len(group_width) == 0:
                    continue
                index = np.array(group_width).argmax()
                if graph[str(t - 1) + str(index)]['width'] <= graph[str(t) + str(i)]['width']:
                    graph[str(t) + str(i)]['width'] = graph[str(t - 1) + str(index)]['width']
        return graph

    def get_path_from_graph(self, graph, nb_arrival_group):
        path = None
        width_tab = []
        for i in range(nb_arrival_group):
            width_tab.append(graph[str(int(self.world.time_horizon // self.world.time_step_res - 1)) + str(i)]['width'])
        # We keep only the possible paths leading to the largest way
        index = np.array(width_tab).argmax()
        indexes = np.where(width_tab == width_tab[index])[0]
        selected_index = None
        for equivalent_index in indexes:
            if 'free_path' in graph[str(
                    int(self.world.time_horizon // self.world.time_step_res - 1)) + str(equivalent_index)]:
                selected_index = equivalent_index
        if selected_index is not None:
            index = selected_index
        if 'free_path' not in graph[str(int(self.world.time_horizon // self.world.time_step_res - 1)) + str(index)]:
            paths_in_rfs_graph = find_paths(graph, '00',
                                            str(int(self.world.time_horizon // self.world.time_step_res - 1)) + str(
                                                index))
            if len(paths_in_rfs_graph) > 0:
                return None
            else:
                return "no way"
        else:
            free_path = graph[str(int(self.world.time_horizon // self.world.time_step_res - 1)) + str(index)][
                'free_path']
            free_index = None
            previous_status = ''
            if len(free_path) == 0:
                print("only one free path")
                free_index = free_path[0]
            else:
                min_dist = float('inf')
                for test_index in free_path:
                    end_point = \
                        graph[str(int(self.world.time_horizon // self.world.time_step_res - 1)) + str(index)]["group"][
                            graph[str(int(self.world.time_horizon // self.world.time_step_res - 1)) + str(index)][
                                'group'][:, 0] == test_index][0][1:3]
                    nearest_waypoint = self.world.map.get_waypoint(carla.Location(x=end_point[0],
                                                                                  y=end_point[1]))
                    nearest_waypoint_point = carla_vector2array_2d(
                        nearest_waypoint.transform.location).round(decimals=2)
                    next_nearest_point = carla_vector2array_2d(
                        nearest_waypoint.next(self.world.global_path_interval)[0].transform.location).round(decimals=2)
                    previous_nearest_point = carla_vector2array_2d(
                        nearest_waypoint.previous(
                            self.world.global_path_interval)[0].transform.location).round(decimals=2)

                    path_vector_next = np.add(next_nearest_point, - nearest_waypoint_point)
                    path_vector_previous = np.add(previous_nearest_point, - nearest_waypoint_point)
                    end_point_vector = np.add(end_point, - nearest_waypoint_point)

                    dot_product = max(float(np.dot(end_point_vector, path_vector_next)),
                                      float(np.dot(end_point_vector, path_vector_previous)))

                    dist = np.linalg.norm(end_point_vector) ** 2 - dot_product ** 2
                    print(test_index, ": ", dist)
                    if dist < self.world.vehicle.bounding_box.extent.y ** 2:
                        print(test_index, "is on a lane")
                        if nearest_waypoint.is_junction:
                            section_hash_tag = nearest_waypoint.junction_id
                        else:
                            section_hash_tag = hash((str(nearest_waypoint.road_id), str(nearest_waypoint.section_id)))
                        lane_hash_tag = hash((str(nearest_waypoint.road_id), str(nearest_waypoint.section_id),
                                              str(nearest_waypoint.lane_id)))
                        alternatives = self.planner.global_paths_alternative[section_hash_tag]
                        if lane_hash_tag in alternatives["alternative_lane"]:
                            status = alternatives["alternative_lane"][lane_hash_tag]["status"]
                            if len(previous_status) == 0:
                                previous_status = status
                                free_index = test_index
                                min_dist = dist
                            else:
                                if status == previous_status and dist < min_dist:
                                    previous_status = status
                                    if free_index == self.world.nb_paths:
                                        continue
                                    free_index = test_index
                                    min_dist = dist
                                else:
                                    if (previous_status[-1] == 'l' and (status[-1] == 'r' or status == 's')) \
                                            or (previous_status[-1] == 'l' and len(status) < len(previous_status)) \
                                            or (previous_status[-1] == 'r' and len(status) > len(previous_status)) \
                                            or (previous_status == 'o' and status != 'o'):
                                        previous_status = status
                                        free_index = test_index
                                        min_dist = dist
                        else:
                            if nearest_waypoint.is_junction:
                                for alt_hash in \
                                        self.planner.global_paths_alternative[section_hash_tag]["alternative_lane"]:
                                    intermediate_lane = self.planner.get_lane_from_hash(section_hash_tag, alt_hash)
                                    if self.planner.local_lane[-1]["status"][-1] == intermediate_lane["status"][0]:
                                        previous_status = intermediate_lane["status"][0]
                                        if free_index == self.world.nb_paths:
                                            continue
                                        free_index = test_index
                                        min_dist = dist
                            else:
                                print("not in alternative lane")
                if free_index != self.world.nb_paths:
                    print("not in the middle")
                print(free_index, "is the best")
            if free_index is None:
                print("no index near a lane we take the middle of the free zone")
                free_path_grouped = []
                previous_index = None
                temp_group = []
                for free_path_index in free_path:
                    if previous_index is None:
                        previous_index = free_path_index
                        temp_group = [free_path_index]
                    else:
                        if free_path_index - previous_index == 1:
                            temp_group.append(free_path_index)
                        else:
                            free_path_grouped.append(temp_group)
                            temp_group = [free_path_index]
                free_path_grouped.append(temp_group)
                free_path_selected_group = None
                max_group_len = 0
                for group in free_path_grouped:
                    group_len = len(group)
                    if group_len > max_group_len:
                        max_group_len = group_len
                        free_path_selected_group = group
                free_index = free_path_selected_group[len(free_path_selected_group) // 2]
            paths_in_rfs_graph = find_paths(graph, '00',
                                            str(int(self.world.time_horizon // self.world.time_step_res - 1)) + str(
                                                index))
            if len(paths_in_rfs_graph) > 0:
                path_in_rfs_graph = paths_in_rfs_graph[0]
            else:
                raise Exception('Is this can append ?')
            if path_in_rfs_graph is not None:
                ris_path = []
                for key in path_in_rfs_graph:
                    if key is '00':
                        ris_path.append(carla_vector2array_2d(self.world.vehicle.get_location()).round(decimals=2))
                    else:
                        if len(graph[key]['group'][
                                   graph[key]['group'][:, 0] == free_index]) > 0:
                            ris_path.append(
                                graph[key]['group'][
                                    graph[key]['group'][:, 0] == free_index][0][1:3]
                            )
                        else:
                            key_t = key[:-1]
                            acc = 0
                            while True:
                                key_variation = key_t + str(acc)
                                if key_variation not in graph:
                                    break
                                if key_variation == key:
                                    acc += 1
                                else:
                                    if len(
                                            graph[key_variation]['group'][
                                                graph[key_variation][
                                                    'group'][:, 0] == free_index]
                                    ) > 0:
                                        ris_path.append(graph[key_variation]['group'][
                                                            graph[key_variation][
                                                                'group'][:, 0] == free_index][
                                                            0][1:3])
                                        break
                                    else:
                                        acc += 1
                path = np.array(ris_path).round(decimals=2)
        return path

    def get_speed_from_profile(self, actual_speed, speed_profile, max_acceleration, max_speed, deceleration):
        speed = [actual_speed]
        if speed_profile == SpeedPlan('speed_up'):
            for t in range(self.world.nb_step + 1):
                speed.append(min(speed[-1] + max_acceleration * self.world.time_step_res, max_speed))
        elif speed_profile == SpeedPlan('keep_speed'):
            for t in range(self.world.nb_step + 1):
                speed.append(speed[-1])
        elif speed_profile == SpeedPlan('brake'):
            for t in range(self.world.nb_step + 1):
                speed.append(max(speed[-1] - deceleration * self.world.time_step_res, actual_speed / 2))
        else:
            raise (Exception("The speed profile - %s - is wrong or not supported yet" % speed_profile))
        return np.array(speed).round(decimals=2)

    def compute_rfs(self):
        interaction_set, free_set, _ = self.world.occupancy_mapper.intersection_rs_circles(eps=self.world.margin)
        if self.world.display:
            for set_slice in free_set:
                for group in set_slice:
                    for point in group:
                        self.world.occupancy_viewer_ris.add_point(np.array([point[1], point[2]]).round(decimals=2),
                                                                  color="#ff00ff",
                                                                  frame=self.world.recorder.current_frame,
                                                                  screens=self.screen,
                                                                  label='reachable \n sets'
                                                                  )
        return interaction_set, free_set

    def get_path_from_ris_wrapper(self, ris, rfs, goal, screen=None):
        ris_polygons = []
        for poly in get_projected_polygons(ris):
            if len(poly) < 3:
                continue
            else:
                ris_polygon = Polygon([tuple(point) for point in poly]).convex_hull
            ris_polygons.append(ris_polygon)
        for line in self.world.occupancy_mapper.static_ris:
            ris_polygons.append(line)
        ris_polygons = clean_obstacles(ris_polygons,
                                       self.world.margin,
                                       self.world.tolerance)
        # if self.world.display:
        #     for poly in ris_polygons:
        #         self.world.occupancy_viewer_ris.add_polygon(
        #             poly,
        #             color='#775000',
        #             frame=self.world.recorder.current_frame,
        #             screens=screen,
        #             label='ris polygons'
        #         )
        for poly in self.world.occupancy_mapper.static_poly[str(
                self.world.recorder.current_frame
        )]:
            ris_polygons.append(poly)
        ris_polygons = clean_obstacles(ris_polygons,
                                       self.world.tolerance,
                                       self.world.tolerance)
        if self.world.display:
            for poly in ris_polygons:
                self.world.occupancy_viewer_ris.add_polygon(
                    poly,
                    color='#775000',
                    frame=self.world.recorder.current_frame,
                    screens=screen,
                    label='ris polygons'
                )
        path = self.get_path_from_rfs_cost(rfs, ris_polygons, goal, screen)
        return path

    def get_path_from_rfs_cost(self, rfs, ris, goal, screen):
        # todo : rfs class could be used all the time
        rfs_reformatted = []
        for rfs_slice in rfs:
            rfs_reformatted.append([])
            for point in rfs_slice[0]:
                rfs_reformatted[-1].append(
                    Set(point[1:3], point[0], get_cost(point[1:3], goal, self.world,
                                                       self.world.scenario_data["apf_params"], extra_polys=ris))
                )
        graph = Graph()
        graph.add_node_from_rfs(rfs_reformatted)
        path = graph.get_apf_path(np.deg2rad(self.world.vehicle.get_transform().rotation.yaw), goal, self.world, screen)
        dist_step = self.world.vehicle_speed_limit * self.world.occupancy_mapper.world.time_step_res
        if dist_step < self.world.global_path_interval:
            dist_step = self.world.global_path_interval
        u_max = np.deg2rad(self.world.occupancy_mapper.vehicle_profile["specific"]["max_steer_angle"])
        wheel_spacing = self.world.occupancy_mapper.vehicle_profile["specific"]["wheel_spacing"]
        realistic_path = get_realistic_path(np.array(path), path[0][:2], path[0][2], dist_step, u_max,
                                            wheel_spacing,
                                            self.world.scenario_data["L"], self.world.visual_horizon)
        return realistic_path[:, :2]
