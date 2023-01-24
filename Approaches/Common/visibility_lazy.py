import json

import numpy as np
import pyvisgraph as vg
import shapely.geometry
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from shapely.ops import cascaded_union

INF = 10000
epsilon = 0.01
# tolerance = 2


def clean_obstacles(obstacles, buffer_size=0, tolerance=0):
    # Merge superposed obstacles
    obs = None
    try:
        polygons = []
        for obs in obstacles:
            poly = obs.buffer(buffer_size)
            polygons.append(poly.simplify(tolerance, preserve_topology=False))
    except Exception as e:
        print(e)
        try:
            polygons = []
            for obs in obstacles:
                poly = Polygon(np.unique(obs, axis=0)).buffer(buffer_size)
                poly = poly.simplify(tolerance, preserve_topology=False)
                if poly.area > tolerance ** 2:
                    polygons.append(poly)
        except Exception as e:
            with open('logs.json', 'w') as f:
                json.dump({"obs": list(obs.exterior.coords)}, f)
            raise e
    merged = cascaded_union(polygons)
    if isinstance(merged, shapely.geometry.multipolygon.MultiPolygon):
        return list(merged)
    elif isinstance(merged, shapely.geometry.polygon.Polygon):
        return [merged]
    else:
        return []


def get_filtered_obstacles(start, dest, obstacles):
    # Check that start and dest are not inside an obstacle
    # And find if there are inside a containingPoly (in interiorPoly)
    containing_poly = None
    interior_poly = None
    for obs in obstacles:
        if obs.intersection(Point(*start)) or obs.intersection(Point(*dest)):
            return None
        if containing_poly and not _poly_box_is_in_poly_box(obs, containing_poly):
            continue
        if not hasattr(obs, 'interiors'):
            continue
        for interior in obs.interiors:
            int_poly = shapely.geometry.polygon.Polygon(interior.coords)
            start_in_int = int_poly.contains(Point(*start))
            dest_in_int = int_poly.contains(Point(*dest))
            if start_in_int != dest_in_int:
                return None
            if start_in_int and dest_in_int:
                containing_poly = obs
                interior_poly = int_poly
                break

    # Gather filtered obstacles
    filtered_obstacles = []
    for obs in obstacles:
        if containing_poly and not _poly_box_is_in_poly_box(obs, containing_poly):
            continue
        if obs == containing_poly:
            box_interior = shapely.geometry.Polygon(
                [
                    (interior_poly.bounds[0], -INF),
                    (INF, -INF),
                    (INF, INF),
                    (interior_poly.bounds[0], INF)
                ]
            )
            opened = box_interior.difference(interior_poly)
            filtered_obstacles.append(opened)
            box_exterior = shapely.geometry.Polygon(
                [
                    (-INF, -INF),
                    (interior_poly.bounds[0]-epsilon, -INF),
                    (interior_poly.bounds[0]-epsilon, INF),
                    (-INF, INF)
                ]
            )
            filtered_obstacles.append(box_exterior)
        else:
            filtered_obstacles.append(obs)
    return filtered_obstacles


def get_holo_path(start, dest, filtered_obstacles, considered_obstacles, considered_obstacles_id, remaining_obstacles,
                  obstacle_ids_to_consider):
    # lazy path planning around obstacles that collides the path
    start_point = vg.Point(*start)
    dest_point = vg.Point(*dest)
    path_is_free = False
    path_vg = None
    g = None
    for obstacleIdToConsider in obstacle_ids_to_consider:
        considered_obstacles_id.append(obstacleIdToConsider)
        considered_obstacles += [p for p in _get_vis_format_polygons(filtered_obstacles[obstacleIdToConsider])]
    while not path_is_free:
        g = vg.VisGraph()
        g.build(considered_obstacles, status=False)
        path_vg = g.shortest_path(start_point, dest_point)
        if path_vg is None:
            return None, None, None, None, None
        path_is_free, colliding_obstacles, colliding_obstacles_id, remaining_obstacles = _get_colliding_obstacles(
            path_vg,
            remaining_obstacles
        )
        considered_obstacles += [p for poly in colliding_obstacles for p in _get_vis_format_polygons(poly)]
        considered_obstacles_id += colliding_obstacles_id

    return [(p.x, p.y) for p in path_vg], g, considered_obstacles, considered_obstacles_id, remaining_obstacles


def _remove_visibility_edge(graph, vertex_point):
    visibility_graph = graph.visgraph
    for point in visibility_graph.get_points():
        if point == vertex_point:
            for edge in visibility_graph[point]:
                visibility_graph.edges.discard(edge)
            del visibility_graph.graph[point]


def _get_vis_format_polygons(polygon):
    if isinstance(polygon, shapely.geometry.multipolygon.MultiPolygon):
        return [_get_vis_format_contour(p.exterior.coords) for p in polygon]
    elif isinstance(polygon, shapely.geometry.polygon.Polygon):
        return [_get_vis_format_contour(polygon.exterior.coords)]
    else:
        return []


def _get_vis_format_contour(points):
    vis_polygon = []
    last_point = None
    first_point = points[0]
    for p in points:
        if last_point is not None and (
                round(first_point[0], 2) == round(p[0], 2) and round(first_point[1], 2) == round(p[1], 2)):
            continue
        if last_point is None or round(last_point[0], 2) != round(p[0], 2) or round(last_point[1], 2) != round(p[1], 2):
            vis_polygon.append(vg.Point(p[0], p[1]))
            last_point = p
    return vis_polygon


def _poly_box_is_in_poly_box(p, ref):
    return p.bounds[0] >= ref.bounds[0] and\
           p.bounds[2] <= ref.bounds[2] and\
           p.bounds[1] >= ref.bounds[1] and\
           p.bounds[3] <= ref.bounds[3]


def _get_colliding_obstacles(path_vg, obstacles):
    colliding_obstacles = []
    colliding_obstacles_id = []
    remaining_obstacles = []
    colliding_any = False
    for polyID, poly in enumerate(obstacles):
        colliding_poly = False
        for i in range(len(path_vg) - 1):
            segment = [(path_vg[i].x, path_vg[i].y), (path_vg[i + 1].x, path_vg[i + 1].y)]
            intersection = shapely.geometry.LineString(segment).intersection(poly).difference(poly.exterior)
            if _simplify_segment(intersection, poly):
                colliding_poly = True
                break
        if colliding_poly:
            colliding_any = True
            colliding_obstacles.append(poly)
            colliding_obstacles_id.append(polyID)
        else:
            remaining_obstacles.append(poly)
    return not colliding_any, colliding_obstacles, colliding_obstacles_id, remaining_obstacles


def _simplify_segment(segment, poly):
    if isinstance(segment, shapely.geometry.multilinestring.MultiLineString):
        return shapely.geometry.MultiLineString(
            [np.round(np.array(s.coords), 2) for s in segment]
        ).difference(poly.exterior)
    elif isinstance(segment, shapely.geometry.linestring.LineString):
        return shapely.geometry.LineString(np.round(np.array(segment.coords), 2)).difference(poly.exterior)
