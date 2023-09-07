import math

import GPAD.Approaches.Common.dijkstra as dijkstra
import numpy as np
import shapely.geometry
from GPAD.Approaches.Common.visibility_lazy import get_filtered_obstacles, get_holo_path

orientationEpsilon = 1


def get_path(start, start_orientation, dest, yaw_rate_radius, obstacles, cached_graph=None, dest_orientation=None):
    key = None
    filtered_obstacles = get_filtered_obstacles(start, dest, obstacles)
    if filtered_obstacles is None:
        return cached_graph, None, None
    considered_obstacles = []
    considered_obstacles_id = []
    remaining_obstacles = filtered_obstacles[:]
    obstacle_ids_to_consider = []
    path_holo, cached_graph, considered_obstacles, considered_obstacles_id, remaining_obstacles = get_holo_path(
        start,
        dest,
        filtered_obstacles,
        considered_obstacles,
        considered_obstacles_id,
        remaining_obstacles,
        obstacle_ids_to_consider
    )
    if path_holo is None:
        return cached_graph, None, None
    orientation_point = (
        start[0] + orientationEpsilon * math.cos(start_orientation),
        start[1] + orientationEpsilon * math.sin(start_orientation)
    )
    path_holo.insert(1, orientation_point)
    if dest_orientation is not None:
        orientation_point = (
            dest[0] + orientationEpsilon * math.cos(dest_orientation),
            dest[1] + orientationEpsilon * math.sin(dest_orientation)
        )
        path_holo.insert(-2, orientation_point)
    yaw_rate_circles_graph, segment_arc_infos = _get_noholo_path(path_holo, yaw_rate_radius, filtered_obstacles)
    circles_order = yaw_rate_circles_graph.shortest_path((-1, -1, 0, 0), (len(path_holo) - 1, 0, -1, -1))
    if circles_order is None:
        return cached_graph, None, path_holo
    if len(circles_order) <= 3:
        key = circles_order[1]
        noholo_path = [np.array(p) for p in segment_arc_infos[key]["segmentFromPrev"]]
        return cached_graph, noholo_path, path_holo
    noholo_path = []
    for cid in range(1, len(circles_order) - 2):
        key = circles_order[cid] + circles_order[cid + 1][-4:]
        noholo_path += [np.array(p) for p in segment_arc_infos[key]["segmentFromPrev"]]
        noholo_path += segment_arc_infos[key]["arc"]["points"]
    noholo_path += [np.array(p) for p in segment_arc_infos[key]["segmentToNext"]]
    return cached_graph, noholo_path, path_holo


def _get_noholo_path(path_holo, yaw_rate_radius, obstacles):
    segment_arc_infos = {}
    yaw_rate_circles_graph = dijkstra.Graph()

    cached_circles = {"yawRateRadius": yaw_rate_radius}
    reached_circles_keys = [(1, s) for s in _get_circles_keys()]

    # Vertices with a circle
    for vertexId in range(1, len(path_holo) - 1):
        reached_diff = []
        left = _get_vertex_left(vertexId, cached_circles, path_holo)
        for c in [s for s in _get_circles_keys() if (vertexId, s) in reached_circles_keys]:
            center = _get_vertex_circle(vertexId, c, cached_circles, path_holo)["center"]
            prev_vertex = vertexId - 1
            next_vertex = vertexId
            still_to_test = 2  # 2: still to test forward, 1: still to test backward, 0: nothing to test
            while still_to_test:
                if still_to_test == 2:
                    next_vertex += 1
                else:
                    prev_vertex -= 1
                    if prev_vertex == 0:
                        break
                if next_vertex == len(path_holo):
                    next_vertex -= 1
                    still_to_test -= 1
                    continue
                if prev_vertex == -1:
                    break

                if prev_vertex == 0:
                    tangent_point = _get_tangent_point(center, yaw_rate_radius, path_holo[0], left)
                    if tangent_point is None:
                        if still_to_test == 2:
                            next_vertex -= 1
                            if next_vertex == vertexId:
                                break
                        still_to_test -= 1
                        continue
                    first_segment = [path_holo[0], tangent_point]
                    if next_vertex == len(path_holo) - 1:
                        tangent_point = _get_tangent_point(center, yaw_rate_radius, path_holo[-1], not left)
                        if tangent_point is not None:
                            last_segment = [tangent_point, path_holo[-1]]
                            arc_points, arc_length = _get_arc_circle(
                                center, yaw_rate_radius, first_segment[1], last_segment[0], left)
                            arc_infos = {
                                "center": center,
                                "radius": yaw_rate_radius,
                                "left": left,
                                "points": arc_points,
                                "length": arc_length
                            }
                            is_free, colliding_infos = _segments_are_free([first_segment, arc_points, last_segment],
                                                                          obstacles)
                            if is_free:
                                reached_diff.append((len(path_holo) - 1, 0))
                                _add_node(vertexId, c, 0, 0, len(path_holo) - 1, 0, first_segment, arc_infos,
                                          last_segment, segment_arc_infos, yaw_rate_circles_graph)
                    else:
                        for nc in [s for s in _get_circles_keys() if (next_vertex, s) not in reached_circles_keys]:
                            next_left = _get_vertex_left(next_vertex, cached_circles, path_holo)
                            next_center = _get_vertex_circle(next_vertex, nc, cached_circles, path_holo)["center"]
                            # Segment from current to next circle
                            if next_left != left:
                                # Build tangent segment between 2 circles
                                dist = math.hypot(next_center[0] - center[0], next_center[1] - center[1])
                                if dist < 2 * yaw_rate_radius:
                                    continue
                                segment_to_next = _get_tangent_segment_between(center, yaw_rate_radius, next_center,
                                                                               yaw_rate_radius, next_left)
                            else:
                                # Build tangent segment outside 2 circles
                                segment_to_next = _get_tangent_segment_outside(center, yaw_rate_radius, next_center,
                                                                               yaw_rate_radius, next_left)
                            arc_points, arc_length = _get_arc_circle(center, yaw_rate_radius, first_segment[1],
                                                                     segment_to_next[0], left)
                            arc_infos = {"center": center, "radius": yaw_rate_radius, "left": left,
                                         "points": arc_points, "length": arc_length}
                            is_free, colliding_infos = _segments_are_free(
                                [first_segment, arc_points, segment_to_next], obstacles)
                            if is_free:
                                reached_diff.append((next_vertex, nc))
                                _add_node(vertexId, c, 0, 0, next_vertex, nc, first_segment, arc_infos, segment_to_next,
                                          segment_arc_infos, yaw_rate_circles_graph)
                else:
                    last_segment = None
                    if next_vertex == len(path_holo) - 1:
                        tangent_point = _get_tangent_point(center, yaw_rate_radius, path_holo[-1], not left)
                        if tangent_point is None:
                            if still_to_test == 2:
                                next_vertex -= 1
                                if next_vertex == vertexId:
                                    break
                            still_to_test -= 1
                            continue
                        last_segment = [tangent_point, path_holo[-1]]
                    for pc in _get_circles_keys():
                        if next_vertex == len(path_holo) - 1:
                            prev_left = _get_vertex_left(prev_vertex, cached_circles, path_holo)
                            prev_center = _get_vertex_circle(prev_vertex, pc, cached_circles, path_holo)["center"]
                            # Segment from previous to current circle
                            if left != prev_left:
                                # Build tangent segment between 2 circles
                                dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
                                if dist < 2 * yaw_rate_radius:
                                    continue
                                segment_from_prev = _get_tangent_segment_between(prev_center, yaw_rate_radius, center,
                                                                                 yaw_rate_radius, left)
                            else:
                                # Build tangent segment outside 2 circles
                                segment_from_prev = _get_tangent_segment_outside(prev_center, yaw_rate_radius, center,
                                                                                 yaw_rate_radius, left)
                            arc_points, arc_length = _get_arc_circle(center, yaw_rate_radius, segment_from_prev[1],
                                                                     last_segment[0], left)
                            arc_infos = {"center": center, "radius": yaw_rate_radius, "left": left,
                                         "points": arc_points, "length": arc_length}
                            is_free, colliding_infos = _segments_are_free([segment_from_prev, arc_points, last_segment],
                                                                          obstacles)
                            if is_free:
                                reached_diff.append((len(path_holo) - 1, 0))
                                _add_node(vertexId, c, prev_vertex, pc, len(path_holo) - 1, 0, segment_from_prev,
                                          arc_infos, last_segment, segment_arc_infos, yaw_rate_circles_graph)
                        else:
                            for nc in [s for s in _get_circles_keys() if (next_vertex, s) not in reached_circles_keys]:
                                prev_left = _get_vertex_left(prev_vertex, cached_circles, path_holo)
                                prev_center = _get_vertex_circle(prev_vertex, pc, cached_circles, path_holo)["center"]
                                next_left = _get_vertex_left(next_vertex, cached_circles, path_holo)
                                next_center = _get_vertex_circle(next_vertex, nc, cached_circles, path_holo)["center"]
                                # Segment from previous to current circle
                                if left != prev_left:
                                    # Build tangent segment between 2 circles
                                    dist = math.hypot(center[0] - prev_center[0], center[1] - prev_center[1])
                                    if dist < 2 * yaw_rate_radius:
                                        continue
                                    segment_from_prev = _get_tangent_segment_between(prev_center, yaw_rate_radius,
                                                                                     center,
                                                                                     yaw_rate_radius, left)
                                else:
                                    # Build tangent segment outside 2 circles
                                    segment_from_prev = _get_tangent_segment_outside(prev_center, yaw_rate_radius,
                                                                                     center,
                                                                                     yaw_rate_radius, left)
                                # Segment from current to next circle
                                if next_left != left:
                                    # Build tangent segment between 2 circles
                                    dist = math.hypot(next_center[0] - center[0], next_center[1] - center[1])
                                    if dist < 2 * yaw_rate_radius:
                                        continue
                                    segment_to_next = _get_tangent_segment_between(center, yaw_rate_radius, next_center,
                                                                                   yaw_rate_radius, next_left)
                                else:
                                    # Build tangent segment outside 2 circles
                                    segment_to_next = _get_tangent_segment_outside(center, yaw_rate_radius, next_center,
                                                                                   yaw_rate_radius, next_left)
                                arc_points, arc_length = _get_arc_circle(center, yaw_rate_radius, segment_from_prev[1],
                                                                         segment_to_next[0], left)
                                arc_infos = {"center": center, "radius": yaw_rate_radius, "left": left,
                                             "points": arc_points, "length": arc_length}
                                is_free, colliding_infos = _segments_are_free(
                                    [segment_from_prev, arc_points, segment_to_next], obstacles)
                                if is_free:
                                    reached_diff.append((next_vertex, nc))
                                    _add_node(vertexId, c, prev_vertex, pc, next_vertex, nc, segment_from_prev,
                                              arc_infos,
                                              segment_to_next, segment_arc_infos, yaw_rate_circles_graph)
        reached_circles_keys += list(set(reached_diff))
    return yaw_rate_circles_graph, segment_arc_infos


def _get_arc_circle(center, radius, p1, p2, direction, epsilon=1):
    theta1 = math.atan2(p1[1] - center[1], p1[0] - center[0])
    theta2 = math.atan2(p2[1] - center[1], p2[0] - center[0])
    diff = theta2 - theta1
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff <= -math.pi:
        diff += 2 * math.pi
    if abs(diff) < 0.1:
        return [], 0
    if not direction and theta2 < theta1:
        theta2 += 2 * math.pi
    if direction and theta2 > theta1:
        theta2 -= 2 * math.pi
    nb_points = max(3, int(radius * abs(theta2 - theta1) / epsilon))
    theta = np.linspace(theta1, theta2, nb_points)
    if theta.size < 2:
        theta = np.array([theta1, theta2])
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    points = list(np.column_stack([x, y]))
    return points, abs(diff * radius)


def _get_tangent_point(circle_center, radius, point, left):
    dx = circle_center[0] - point[0]
    dy = circle_center[1] - point[1]
    dist = math.hypot(dx, dy)
    if dist <= radius:
        return None
    a = math.asin(radius / dist)
    o = math.atan2(dy, dx)
    if left:
        return circle_center[0] - radius * math.sin(o + a), circle_center[1] + radius * math.cos(o + a)
    else:
        return circle_center[0] + radius * math.sin(o - a), circle_center[1] - radius * math.cos(o - a)


def _get_tangent_segment_between(circle1_center, radius1, circle2_center, radius2, left):
    # https://stackoverflow.com/questions/27970185/find-line-that-is-tangent-to-2-given-circles
    dx = circle2_center[0] - circle1_center[0]
    dy = circle2_center[1] - circle1_center[1]
    d = math.hypot(dx, dy)
    if radius1 == 0 and radius2 == 0:
        h1 = d / 2
        h2 = d / 2
    # TODO : Special case needed if only one radius is 0. But the overall code is not compatible with this case for now
    else:
        m = radius1 / radius2
        h1 = m * d / (1 + m)  # distance from circle1Center to middlePoint
        h2 = d / (1 + m)  # distance from circle2Center to middlePoint
    middle_point_x = (h2 * circle1_center[0] + h1 * circle2_center[0]) / d
    middle_point_y = (h2 * circle1_center[1] + h1 * circle2_center[1]) / d
    point1 = _get_tangent_point(circle1_center, radius1, (middle_point_x, middle_point_y), left)  # @HERE: no not
    point2 = _get_tangent_point(circle2_center, radius2, (middle_point_x, middle_point_y), left)  # @HERE: no not
    return [point1, point2]


def _get_tangent_segment_outside(circle1_center, radius1, circle2_center, radius2, left):
    dx = circle2_center[0] - circle1_center[0]
    dy = circle2_center[1] - circle1_center[1]
    # d = math.hypot(dx, dy)
    theta = math.atan2(dy, dx)

    if radius1 == radius2:
        alpha = theta + (1 if left else -1) * math.pi / 2
        point1 = [circle1_center[0] + radius1 * math.cos(alpha), circle1_center[1] + radius1 * math.sin(alpha)]
        point2 = [circle2_center[0] + radius1 * math.cos(alpha), circle2_center[1] + radius1 * math.sin(alpha)]
        return [point1, point2]
    else:
        # @TODO : compute intersection point O
        # with dist(O, circle1Center) = d*radius1/(radius2-radius1)
        # then extract the tangent segment
        # but useless for now anyway
        pass


def _segments_are_free(segments, obstacles):
    for obsId, obs in enumerate(obstacles):
        for segment in segments:
            if _segment_polygon_open_intersection(segment, obs):
                return False, obsId
    return True, None


def _segment_polygon_open_intersection(segment, poly):
    intersection = shapely.geometry.LineString(segment).intersection(poly).difference(poly.exterior)
    return _simplify_segment(intersection, poly)


def _simplify_segment(segment, poly):
    if isinstance(segment, shapely.geometry.multilinestring.MultiLineString):
        return shapely.geometry.MultiLineString([np.round(np.array(s.coords), 2) for s in segment]).difference(
            poly.exterior)
    elif isinstance(segment, shapely.geometry.linestring.LineString):
        return shapely.geometry.LineString(np.round(np.array(segment.coords), 2)).difference(poly.exterior)


def _add_node(vertex_id, c, prev_vertex, pc, next_vertex, nc, segment_from_prev, arc_infos, segment_to_next,
              segment_arc_infos, yaw_rate_circles_graph):
    dist = 0
    if segment_from_prev:
        length_from_prev = math.hypot(segment_from_prev[0][0] - segment_from_prev[1][0],
                                      segment_from_prev[0][1] - segment_from_prev[1][1])
        dist += length_from_prev if pc == 0 else length_from_prev / 2
    if segment_to_next:
        length_to_next = math.hypot(segment_to_next[0][0] - segment_to_next[1][0],
                                    segment_to_next[0][1] - segment_to_next[1][1])
        dist += length_to_next if nc == 0 else length_to_next / 2
    if arc_infos:
        dist += arc_infos["length"]
    if vertex_id == -1:
        yaw_rate_circles_graph.add_edge((-1, -1, 0, 0), (0, 0, next_vertex, 0), dist)
        yaw_rate_circles_graph.add_edge((0, 0, next_vertex, 0), (next_vertex, 0, -1, -1), 0)
        key = (prev_vertex, pc, next_vertex, nc)
        segment_arc_infos[key] = {"segmentFromPrev": segment_from_prev, "arc": arc_infos,
                                  "segmentToNext": segment_to_next}
    else:
        if pc == 0:
            yaw_rate_circles_graph.add_edge((-1, -1, 0, 0), (0, 0, vertex_id, c), 0)
        if nc == 0:
            yaw_rate_circles_graph.add_edge((vertex_id, c, next_vertex, 0), (next_vertex, 0, -1, -1), 0)

        yaw_rate_circles_graph.add_edge((prev_vertex, pc, vertex_id, c), (vertex_id, c, next_vertex, nc), dist)
        key = (prev_vertex, pc, vertex_id, c, vertex_id, c, next_vertex, nc)
        segment_arc_infos[key] = {"segmentFromPrev": segment_from_prev, "arc": arc_infos,
                                  "segmentToNext": segment_to_next}


def _get_vertex_left(vertex_id, cached_circles, path_holo):
    return _get_circle_cache(vertex_id, cached_circles, path_holo)["left"]


def _get_vertex_circle(vertex_id, c, cached_circles, path_holo):
    return _get_circle_cache(vertex_id, cached_circles, path_holo)["circles"][c]


def _get_circles_keys():
    return ["a", "b", "c"]


def _get_circle_cache(vertex_id, cached_circles, path_holo):
    if vertex_id in cached_circles:
        return cached_circles[vertex_id]
    else:
        t_prev = math.atan2(path_holo[vertex_id - 1][1] - path_holo[vertex_id][1],
                            path_holo[vertex_id - 1][0] - path_holo[vertex_id][0])
        t_next = math.atan2(path_holo[vertex_id + 1][1] - path_holo[vertex_id][1],
                            path_holo[vertex_id + 1][0] - path_holo[vertex_id][0])
        t_diff = t_next - t_prev
        while t_diff > math.pi:
            t_diff -= 2 * math.pi
        while t_diff <= -math.pi:
            t_diff += 2 * math.pi
        left = t_diff > 0
        circles = {
            "a": {
                "center": _get_center_c1(path_holo[vertex_id], t_prev, t_diff,
                                         cached_circles["yawRateRadius"])},
            "b": {
                "center": _get_center_c2(path_holo[vertex_id], t_next, t_diff,
                                         cached_circles["yawRateRadius"])},
            "c": {
                "center": _get_center_c3(path_holo[vertex_id], t_next, t_diff, cached_circles["yawRateRadius"])}
        }
        cached_circles[vertex_id] = {"left": left, "circles": circles}
        return cached_circles[vertex_id]


def _get_center_c1(vertex, t_prev, t_diff, radius, epsilon=0):
    theta_c1 = t_prev + (math.pi / 2 if t_diff > 0 else -math.pi / 2)
    return vertex[0] + (radius - epsilon) * math.cos(theta_c1), vertex[1] + (radius - epsilon) * math.sin(theta_c1)


def _get_center_c2(vertex, t_next, t_diff, radius, epsilon=0):
    theta_c2 = t_next - t_diff / 2
    return vertex[0] + (radius - epsilon) * math.cos(theta_c2), vertex[1] + (radius - epsilon) * math.sin(theta_c2)


def _get_center_c3(vertex, t_next, t_diff, radius, epsilon=0):
    theta_c3 = t_next + (math.pi / 2 if t_diff < 0 else -math.pi / 2)
    return vertex[0] + (radius - epsilon) * math.cos(theta_c3), vertex[1] + (radius - epsilon) * math.sin(theta_c3)
