import json
import math

import numpy as np
import shapely.geometry
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
                                #  beware time is int and must be multiplied by a time step
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
                                #  beware time is int and must be multiplied by a time step
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
            # the radial line intersects at one point only
            perimeter.append(inter)

        if inter.type == "GeometryCollection":
            # the radial line doesn't intersect, so add the end point of the line
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
        # the forward point distance smooth the path_point, usually we take wheel spacing, but I set to other value
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


def get_out_path(path, s):
    prc_point = None
    path_dist = [0]
    dist = 0
    for index, point in enumerate(path):
        if index == 0:
            prc_point = point
        else:
            dist += np.linalg.norm(point - prc_point)
            path_dist.append(dist)
            prc_point = point
    index = 0
    out_path = [path[0]]
    for si in s:
        if si[1] == 0:
            continue
        while si[1] > path_dist[index + 1]:
            index += 1
            if index >= len(path_dist) - 1:
                return out_path
        dist = si[1] - path_dist[index]
        vector = path[index + 1] - path[index]
        vector /= np.linalg.norm(vector)
        point = path[index] + dist * vector
        out_path.append(point)
    return out_path
