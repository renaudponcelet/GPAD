import math
import re

import carla
import numpy as np
from .utils import (get_visible_exteriors_from_polys, get_index_offset, find_nearest_vector,
                    get_diff_angle, get_realistic_path)


class Set:
    def __init__(self, point, index, cost):
        self.point = point
        self.index = index
        self.cost = cost


class Node:

    def __init__(self, point, index, cost, time):
        self.index = index
        self.point = point
        self.t = time
        self.cost = cost

    def dist(self, point):
        return np.linalg.norm(np.add(self.point, - point))


class Graph:

    def __init__(self):
        self.node_list = []
        self.arrival_node_list = []
        self.start_node = None
        self.max_time = 0
        self.min_cost = None
        self.max_cost = None

    def nearest(self, point):
        min_dist = float("inf")
        nearest_node = None
        for node in self.node_list:
            dist = node.dist(point)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        return nearest_node

    def add_node(self, point, index, cost, time, arrival=False):
        if time > self.max_time:
            self.max_time = time
        self.node_list.append(Node(point, index, cost, time))
        if arrival:
            self.arrival_node_list.append(self.node_list[-1])
        if time == 0:
            self.start_node = self.node_list[-1]
        if self.min_cost is None:
            self.min_cost = cost
        elif cost < self.min_cost:
            self.min_cost = cost
        if self.max_cost is None:
            self.max_cost = cost
        elif cost > self.max_cost:
            self.max_cost = cost

    def add_nodes_by_slice(self, rfs_slice, time, arrival=False):
        for rfs in rfs_slice:
            self.add_node(rfs.point, rfs.index, rfs.cost, time, arrival=arrival)

    def add_node_from_rfs(self, rfs):
        for time, rfs_slice in enumerate(rfs):
            self.add_nodes_by_slice(
                rfs_slice=rfs_slice,
                time=time,
                arrival=time == len(rfs)
            )

    def get_apf_path(self, start_heading, destination, world, screen=None):
        if world.display is True and screen is not None:
            for node in self.node_list:
                color = str(hex(int(255 * (node.cost - self.min_cost) / self.max_cost)))
                color = color[color.find('x') + 1:]
                for i in range(6 - len(color)):
                    color = "0" + color
                color = "#" + color
                world.planner.world.occupancy_viewer_ris.add_point(
                    node.point,
                    color=color,
                    frame=world.planner.world.recorder.current_frame,
                    label="apf node",
                    screens=screen
                )
        if start_heading > 2 * math.pi:
            start_heading -= 2 * math.pi
        if start_heading <= 0:
            start_heading += 2 * math.pi
        step_dist = max(world.time_step_res * world.vehicle_speed_limit, world.global_path_interval)
        ego_vehicle_orientation = start_heading
        current_point = self.start_node.point
        path = [[current_point[0], current_point[1], start_heading]]
        t = 0
        while len(path) < world.visual_horizon / step_dist and t < (self.max_time - 1):
            apf_vector = np.array([0, 0])
            flag = True
            min_dist_by_t = {}
            for node in self.node_list:
                if node.t > t:
                    node_vector = node.point - current_point
                    dist = np.linalg.norm(node_vector)
                    if str(node.t) not in min_dist_by_t:
                        min_dist_by_t[str(node.t)] = dist
                    else:
                        if dist < min_dist_by_t[str(node.t)]:
                            min_dist_by_t[str(node.t)] = dist
            time_pre = [int(key) for key in min_dist_by_t if min_dist_by_t[key] > world.scenario_data["L"]]
            if len(time_pre) > 0:
                time = min(time_pre)
            else:
                time = None
            for node in self.node_list:
                if time is not None and time + 2 >= node.t > time:
                    flag = False
                    node_vector = node.point - current_point
                    dist = np.linalg.norm(node_vector)
                    apf_vector = np.add(apf_vector, node_vector / (dist ** 3 * (node.cost - self.min_cost + 1)))
            if flag:
                apf_vector = np.array([destination[0] - current_point[0], destination[1] - current_point[1]]).round(
                    decimals=2)
            apf_vector_orientation = np.arctan2(apf_vector[1], apf_vector[0])
            if apf_vector_orientation <= 0:
                apf_vector_orientation += 2 * math.pi
            if apf_vector_orientation > 2 * math.pi:
                apf_vector_orientation -= 2 * math.pi
            diff_angle = apf_vector_orientation - ego_vehicle_orientation
            if diff_angle >= math.pi:
                diff_angle -= 2 * math.pi
            if diff_angle < - math.pi:
                diff_angle += 2 * math.pi
            max_diff_angle = get_diff_angle(
                np.deg2rad(world.occupancy_mapper.vehicle_profile["specific"]["max_steer_angle"]),
                step_dist,
                world.occupancy_mapper.vehicle_profile["specific"]["wheel_spacing"])
            if np.linalg.norm(diff_angle) > max_diff_angle:
                diff_angle = np.sign(diff_angle) * max_diff_angle
            ego_vehicle_orientation += diff_angle
            if ego_vehicle_orientation > 2 * math.pi:
                ego_vehicle_orientation -= 2 * math.pi
            if ego_vehicle_orientation <= 0:
                ego_vehicle_orientation += 2 * math.pi
            heading_vector = np.array([math.cos(ego_vehicle_orientation), math.sin(ego_vehicle_orientation)])
            speed = step_dist * heading_vector
            current_point = np.add(current_point, speed)
            nearest_node = self.nearest(current_point)
            last_t = t
            t = nearest_node.t
            if t < last_t:
                t = last_t
            path.append([current_point[0], current_point[1], ego_vehicle_orientation])
        return path


def array2carla_location(array):
    return carla.Location(x=array[0], y=array[1], z=array[2])


def carla_vector2array_3d(location):
    return np.array([location.x, location.y, location.z])


def carla_vector2array_2d(location):
    return np.array([location.x, location.y])


def inverse(carla_transform):
    location = carla_transform.location
    inverse_location = carla.Location(x=-location.x, y=-location.y, z=-location.z)
    rotation = carla_transform.rotation
    inverse_rotation = carla.Rotation(yaw=-rotation.yaw, pitch=-rotation.pitch, roll=-rotation.roll)
    return carla.Transform(location=inverse_location), carla.Transform(rotation=inverse_rotation)


def get_image(rec):
    array = np.frombuffer(rec.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (rec.height, rec.width, 4))
    array = array[:, :, :3]
    return array[:, :, ::-1]


def pixel_in_image(pixel, rec):
    return 0 <= int(round(pixel[0])) < rec["depth"].width and \
           0 <= int(round(pixel[1])) < rec["depth"].height and pixel[2] > 0


def pixel_on_image_edge(pixel, rec):
    return 0 == int(round(pixel[0])) or int(round(pixel[0])) == rec["depth"].width - 1 or \
           0 == int(round(pixel[1])) or int(round(pixel[1])) == rec["depth"].height - 1


def point_is_occluded(pixel, rec, tolerance=1):
    depth = get_depth_in_meters(get_image(rec["depth"])[int(round(pixel[1]))][int(round(pixel[0]))])
    diff_depth = depth - pixel[2]
    return diff_depth < -tolerance


def dist_location_2d(location1, location2):
    return math.sqrt((location1.x - location2.x) ** 2 + (location1.y - location2.y) ** 2)


def get_depth_in_meters(depth):
    normalized = (depth[0] + depth[1] * 256 + depth[2] * 256 * 256) / (256 * 256 * 256 - 1)
    return 1000 * normalized


def get_static_path(path, goal_location, local_lane, world):
    local_lane_point = None
    u_max = np.deg2rad(world.occupancy_mapper.vehicle_profile["specific"]["max_steer_angle"])
    wheel_spacing = world.occupancy_mapper.vehicle_profile["specific"]["wheel_spacing"]
    precedent_status = None
    for section in local_lane:
        if precedent_status is None:
            precedent_status = section["status"]
        elif precedent_status[-1] != section["status"][0]:
            break
        else:
            precedent_status = section["status"]
        if local_lane_point is None:
            local_lane_point = array_wp2nd_array(section["lane"]).round(decimals=2)
        else:
            local_lane_point = np.vstack((local_lane_point, array_wp2nd_array(section["lane"]).round(decimals=2)))
    dist_step = world.vehicle_speed_limit * world.occupancy_mapper.world.time_step_res
    if dist_step < world.global_path_interval:
        dist_step = world.global_path_interval
    new_path = [path[0]]
    precedent_point_on_lane = False
    first_index_on_lane = None
    if len(path) > 1:
        precedent_point = path[0]
        precedent_index = 0
        while len(new_path) < world.visual_horizon * world.global_path_interval:
            if precedent_point_on_lane:
                break
            else:
                idx, offset = get_index_offset(
                    np.array(path).round(decimals=2)[:, :2], precedent_point[:2], world, dist_step)

                if idx + offset < len(path):
                    precedent_point = path[idx + offset]
                    indexes = find_nearest_vector(local_lane_point, precedent_point[:2])
                    lane_vector = np.add(local_lane_point[max(indexes)], - local_lane_point[min(indexes)])
                    point_vector = np.add(precedent_point[:2], - local_lane_point[min(indexes)])
                    dot_product = np.dot(point_vector, lane_vector)
                    if dot_product < 0:
                        dot_product *= - 1
                        index = min(indexes)
                    elif dot_product > np.linalg.norm(lane_vector):
                        lane_vector = np.add(local_lane_point[max(indexes) + 1], - local_lane_point[min(indexes) + 1])
                        point_vector = np.add(precedent_point[:2], - local_lane_point[min(indexes) + 1])
                        dot_product = np.dot(point_vector, lane_vector)
                        if dot_product < 0:
                            dot_product *= -1
                            index = min(indexes) + 1
                        else:
                            index = max(indexes) + 1
                    else:
                        index = max(indexes)
                    dist_to_lane = np.linalg.norm(point_vector) ** 2 - dot_product ** 2
                    if dist_to_lane > world.vehicle.bounding_box.extent.y ** 2:
                        precedent_point_on_lane = False
                        first_index_on_lane = None
                    else:
                        if first_index_on_lane is None:
                            first_index_on_lane = index
                        precedent_point_on_lane = True
                        if index < precedent_index:
                            break
                        if index > 0:
                            last_vector = np.add(local_lane_point[index][:2], - local_lane_point[index - 1][:2])
                            heading = np.arctan2(last_vector[1], last_vector[0])
                        else:
                            heading = precedent_point[2]
                        if heading > 2 * math.pi:
                            heading -= 2 * math.pi
                        if heading <= 0:
                            heading += 2 * math.pi
                        precedent_index = index
                        precedent_point = [local_lane_point[index][0], local_lane_point[index][1], heading]
                    new_path.append(precedent_point)
                else:
                    break
        path = np.array(new_path)
    else:
        last_point_path = np.array(path)[-1]
        last_location = last_point_path[:2]
        indexes = find_nearest_vector(local_lane_point, last_location)
        lane_vector = np.add(local_lane_point[max(indexes)], - local_lane_point[min(indexes)])
        point_vector = np.add(last_location, - local_lane_point[min(indexes)])
        dot_product = np.dot(point_vector, lane_vector)
        if dot_product < 0:
            dot_product *= - 1
        elif dot_product > np.linalg.norm(lane_vector):
            lane_vector = np.add(local_lane_point[max(indexes) + 1], - local_lane_point[min(indexes) + 1])
            point_vector = np.add(last_location, - local_lane_point[min(indexes) + 1])
            dot_product = np.dot(point_vector, lane_vector)
        dist_to_lane = np.linalg.norm(point_vector) ** 2 - dot_product ** 2
        if dist_to_lane > world.vehicle.bounding_box.extent.y ** 2:
            precedent_point_on_lane = False
        else:
            precedent_point_on_lane = True
    # # Check if path is close to static path
    last_point_path = np.array(path)[-1]
    last_location = last_point_path[:2]

    if not precedent_point_on_lane:
        fake_static_path = [last_point_path]
        fake_speed_plan = [world.global_path_interval / world.time_step_res]
        for t in range(int(world.time_horizon // world.time_step_res) - len(path)):
            fake_static_path.append(np.add(
                fake_static_path[-1],
                dist_step * np.array([
                    np.cos(fake_static_path[-1][2]),
                    np.sin(fake_static_path[-1][2]),
                    0
                ])
            ))
            fake_speed_plan.append(fake_speed_plan[-1])
        rs, static_ris = world.occupancy_mapper.generator.run(
            np.array(fake_static_path), 6 * world.vehicle.bounding_box.extent.y, fake_speed_plan,
            blocked=world.planner.blocked, nb_step=int(world.time_horizon // world.time_step_res) - len(path)
        )
        rfs = []
        for t in range(len(rs)):
            rfs.append([])
            rfs[t].append([])
            for i in range(len(rs[t])):
                rfs[t][0].append(rs[t][i][0])
        rfs_reformatted = []
        for rfs_slice in rfs:
            rfs_reformatted.append([])
            for point in rfs_slice[0]:
                rfs_reformatted[-1].append(
                    Set(point[1:3], point[0], get_cost(
                        point[1:3], goal_location, world, world.scenario_data["apf_params"]
                    ))
                )
        graph = Graph()
        graph.add_node_from_rfs(rfs_reformatted)
        apf_path = graph.get_apf_path(last_point_path[2], goal_location, world)
        # print("apf path")
        # apf_path = get_apf_path(last_point_path, last_point_path[2], goal_location, world, world.apf_params)
        path = np.concatenate((path, apf_path), axis=0)
        realistic_path = get_realistic_path(path, path[0][:2], path[0][2], dist_step, u_max, wheel_spacing,
                                            world.scenario_data["L"], world.visual_horizon)
        if world.display:
            world.occupancy_viewer_ris.add_line(list(realistic_path), color='#80dd00',
                                                frame=world.recorder.current_frame,
                                                screens=[0, 1, 2, 3],
                                                label='apf path'
                                                )
        return realistic_path
    else:
        while len(path) < int(world.occupancy_mapper.world.time_horizon // world.occupancy_mapper.world.time_step_res):
            idx, offset = get_index_offset(local_lane_point, last_location, world,
                                           dist_step + world.global_path_interval)
            if idx + offset >= len(local_lane_point):
                realistic_path = get_realistic_path(np.array(path), path[0][:2], path[0][2], dist_step, u_max,
                                                    wheel_spacing, world.scenario_data["L"], world.visual_horizon)
                if world.display:
                    world.occupancy_viewer_ris.add_line(list(realistic_path), color='#80dd00',
                                                        frame=world.recorder.current_frame,
                                                        screens=[0, 1, 2, 3],
                                                        label='path'
                                                        )
                return realistic_path
            last_location = np.array(local_lane_point[idx + offset])
            path_point = [last_location[0], last_location[1], 0]
            path = np.vstack((path, path_point))
        realistic_path = get_realistic_path(np.array(path), path[0][:2], path[0][2], dist_step, u_max,
                                            wheel_spacing,
                                            world.scenario_data["L"], world.visual_horizon)
        if world.display:
            world.occupancy_viewer_ris.add_line(list(realistic_path), color='#80dd00',
                                                frame=world.recorder.current_frame,
                                                screens=[0, 1, 2, 3],
                                                label='path'
                                                )
        return realistic_path


def get_cost(point, destination, world, k, extra_polys=None, screens=None):
    if extra_polys is not None:
        polys = world.occupancy_mapper.static_map + extra_polys
    else:
        polys = world.occupancy_mapper.static_map
    repulsive_points = get_visible_exteriors_from_polys(point[:2], polys,
                                                        world.visual_horizon, world, screens=screens)
    x = point[0]
    y = point[1]

    # Compute distance from goal
    goal_vector = np.array([destination[0] - x, destination[1] - y]).round(decimals=2)
    goal_cost = np.linalg.norm(goal_vector)

    # Compute cumulative distance from walls
    walls_distance_tab = []
    for repulsive_point, normal in repulsive_points:
        walls_vector = np.add([x, y], - repulsive_point)
        walls_distance_tab.append(np.linalg.norm(walls_vector))
    wall_distance = min(walls_distance_tab)
    walls_cost = - wall_distance

    # Compute distance from nearest lane
    ego_projection_on_virtual_lane = world.map.get_waypoint(carla.Location(x=x, y=y))
    virtual_lane_apf_vector = np.array(
        [ego_projection_on_virtual_lane.transform.location.x - x,
         ego_projection_on_virtual_lane.transform.location.y - y]
    ).round(decimals=2)
    lane_cost = np.linalg.norm(virtual_lane_apf_vector)

    # Compute distance from last path
    if world.planner.path is not None:
        index = max(find_nearest_vector(world.planner.path, point[:2]))
        last_path_vector = np.array(
            [world.planner.path[index][0] - x,
             world.planner.path[index][1] - y]
        ).round(decimals=2)
        path_cost = np.linalg.norm(last_path_vector)
    else:
        path_cost = 0

    point_cost = (
            k[0] * goal_cost
            + k[1] * walls_cost
            + k[2] * lane_cost
            + k[3] * path_cost
    )
    return point_cost


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def array_wp2nd_array(array, route_mode=False):
    array_point = []
    for step in array:
        if route_mode:
            waypoint_location = step[0].transform.location
        else:
            waypoint_location = step.transform.location
        array_point.append([waypoint_location.x, waypoint_location.y])
    return np.array(array_point)


def get_nearest_tr(transform_list, location):
    min_dist = float('inf')
    nearest_tr = None
    for tr in transform_list:
        dist = dist_location_2d(tr.location, location)
        if dist < min_dist:
            nearest_tr = tr
            min_dist = dist
    return nearest_tr


def get_bounding_box_shape_circles(*args):
    nb_circles = 2
    if len(args) == 1:
        extent_x = args[0].extent.x
        extent_y = args[0].extent.y
    elif len(args) == 2:
        extent_x = args[0]
        extent_y = args[1]
    elif len(args) == 3:
        extent_x = args[0]
        extent_y = args[1]
        nb_circles = args[2]
    else:
        raise AttributeError
    radius = math.sqrt((extent_x / nb_circles) ** 2 / 4 + extent_y ** 2)
    offset = extent_x / nb_circles
    shape = []
    if nb_circles % 2 == 0:
        cum_offset = offset / 2
        for i in range(int(nb_circles / 2)):
            shape.append([cum_offset, 0, 0])
            cum_offset += offset
        shape = np.array(shape)
        shape = np.vstack((-shape[::-1], shape))
    else:
        cum_offset = offset
        for i in range(int(nb_circles / 2)):
            shape.append([cum_offset, 0, 0])
            cum_offset += offset
        shape = np.array(shape)
        shape = np.vstack((-shape[::-1], [0, 0, 0], shape))
    shape = shape.round(decimals=2)
    return shape, radius


def world2pixel(rs_world, last_rec):
    sensor_transform = last_rec["depth"].transform
    screen_transform_yaw = carla.Transform(carla.Location(), carla.Rotation(yaw=90))
    screen_transform_roll = carla.Transform(carla.Location(), carla.Rotation(roll=-90))
    k = np.identity(3)
    k[0, 2] = last_rec["depth"].width / 2.0
    k[1, 2] = last_rec["depth"].height / 2.0
    k[0, 0] = k[1, 1] = last_rec["depth"].width / (
            2.0 * np.tan(last_rec["depth"].fov * math.pi / 360.0)
    )
    sensor_transform_inv_loc, sensor_transform_inv_rot = inverse(sensor_transform)
    rs_car = np.copy(rs_world)
    for t, rs_slice in enumerate(rs_world):
        for i, point in enumerate(rs_slice):
            point_3d = carla.Location(x=point[0], y=point[1], z=point[2])
            point_3d_transform_loc = sensor_transform_inv_loc.transform(point_3d)
            point_3d_transform = sensor_transform_inv_rot.transform(point_3d_transform_loc)
            rs_car[t][i] = np.array([
                point_3d_transform.x,
                point_3d_transform.y,
                point_3d_transform.z
            ])
    rs_screen = np.copy(rs_car)
    for t, rs_slice in enumerate(rs_car):
        for i, point in enumerate(rs_slice):
            point_3d = carla.Location(x=point[0], y=point[1], z=point[2])
            point_3d_transform = screen_transform_yaw.transform(point_3d)
            point_3d_transform = screen_transform_roll.transform(point_3d_transform)
            rs_screen[t][i] = np.array([
                - point_3d_transform.x,
                point_3d_transform.y,
                point_3d_transform.z
            ])
    rs_image = np.copy(rs_screen)
    for t, rs_slice in enumerate(rs_screen):
        for i, point in enumerate(rs_slice):
            rs_image[t][i] = np.dot(k, point)
            rs_image[t][i][0] = rs_image[t][i][0] / rs_image[t][i][2]
            rs_image[t][i][1] = rs_image[t][i][1] / rs_image[t][i][2]
    return rs_image


def pixel2world(pixel, last_rec):
    k = np.identity(3)
    k[0, 2] = last_rec["depth"].width / 2.0
    k[1, 2] = last_rec["depth"].height / 2.0
    k[0, 0] = k[1, 1] = last_rec["depth"].width / (
            2.0 * np.tan(last_rec["depth"].fov * math.pi / 360.0)
    )
    depth_image = get_image(last_rec["depth"])
    depth = depth_image[int(round(pixel[1]))][int(round(pixel[0]))]
    in_meters = get_depth_in_meters(depth)
    image_point = np.linalg.inv(k).dot(np.array([pixel[0], pixel[1], 1])) * in_meters
    image_point = np.array([-image_point[0], image_point[1], image_point[2]])
    image_point = array2carla_location(image_point)
    screen_transform = carla.Transform(carla.Location(), carla.Rotation(roll=90, yaw=-90))
    image_point = screen_transform.transform(image_point)
    sensor_transform = last_rec["depth"].transform
    image_point = sensor_transform.transform(image_point)
    return image_point


def lateral_shift(transform, shift):
    """Makes a lateral shift of the forward vector of a transform"""
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()
