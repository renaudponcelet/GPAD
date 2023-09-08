import hashlib
import json

import carla
import numpy as np
from ....Common.Utils.carla_utils import dist_location_2d, pixel_in_image, carla_vector2array_3d, \
    point_is_occluded, carla_vector2array_2d
from ....Common.Utils.carla_utils import world2pixel
from ....Common.Utils.utils import dist


class VirtualVehicleGenerator:
    name = ""

    def __init__(self, world):
        self.world = world
        self.heights_check_points = [0, 0.5, 1]
        self.waypoints = None

    # def set_waypoint_to_giveaway(self, global_path):
    #     waypoints_to_giveaway = []
    #     path_waypoint_in_junction = None
    #     intersecting_junctions_tab = []
    #     intersection_lane = None
    #     intersection_linestring = None
    #     start_junction = None
    #     dest_junction = None
    #     hash_tag_ignored = []
    #     critic_start_waypoint = []
    #     for step in global_path:
    #         waypoint = step[0]
    #         hash_tag_ignored.append(hash(
    #             (str(waypoint.road_id), str(waypoint.section_id), str(waypoint.lane_id))
    #         ))
    #         if waypoint.is_intersection:
    #             if path_waypoint_in_junction is None:
    #                 path_waypoint_in_junction = hash(
    #                     (str(waypoint.road_id), str(waypoint.section_id), str(waypoint.lane_id))
    #                 )
    #                 break
    #     for waypoints_topology in self.world.topology:
    #         start_waypoint = waypoints_topology[0]
    #         dest_waypoint = waypoints_topology[1]
    #         start_waypoint_tag = hash(
    #             (str(start_waypoint.road_id), str(start_waypoint.section_id), str(start_waypoint.lane_id))
    #         )
    #         dest_waypoint_tag = hash(
    #             (str(dest_waypoint.road_id), str(dest_waypoint.section_id), str(dest_waypoint.lane_id))
    #         )
    #         if start_waypoint_tag == path_waypoint_in_junction or dest_waypoint_tag == path_waypoint_in_junction and \
    #                 np.random.choice(start_waypoint.next(self.world.global_path_interval)).is_intersection:
    #             if intersection_lane is None:
    #                 intersection_linestring = LineString([
    #                     (waypoints_topology[0].transform.location.x, waypoints_topology[0].transform.location.y),
    #                     (waypoints_topology[1].transform.location.x, waypoints_topology[1].transform.location.y)
    #                 ])
    #                 start_junction = np.array([
    #                     waypoints_topology[0].transform.location.x,
    #                     waypoints_topology[0].transform.location.y
    #                 ]).round(decimals=2)
    #                 dest_junction = np.array([
    #                     waypoints_topology[1].transform.location.x,
    #                     waypoints_topology[1].transform.location.y
    #                 ]).round(decimals=2)
    #                 break
    #     for waypoints_topology in self.world.topology:
    #         waypoints_topology_tag = hash((
    #             str(waypoints_topology[0].road_id),
    #             str(waypoints_topology[0].section_id),
    #             str(waypoints_topology[0].lane_id)
    #         ))
    #         if waypoints_topology_tag == path_waypoint_in_junction:
    #             continue
    #         test_linestring = LineString([
    #             (waypoints_topology[0].transform.location.x, waypoints_topology[0].transform.location.y),
    #             (waypoints_topology[1].transform.location.x, waypoints_topology[1].transform.location.y)
    #         ])
    #         if intersection_linestring.intersects(test_linestring):
    #             intersecting_junctions_tab.append(waypoints_topology_tag)
    #             for waypoints_topology_2 in self.world.topology:
    #                 if (waypoints_topology_2[1].transform.location.x, waypoints_topology_2[1].transform.location.y) \
    #                         == (waypoints_topology[0].transform.location.x, waypoints_topology[0].transform.location.y):
    #                     waypoints_topology_tag_2 = hash((
    #                         str(waypoints_topology_2[0].road_id),
    #                         str(waypoints_topology_2[0].section_id),
    #                         str(waypoints_topology_2[0].lane_id)
    #                     ))
    #                     intersecting_junctions_tab.append(waypoints_topology_tag_2)
    #                     critic_start_waypoint.append(waypoints_topology_2[0])
    #     intersecting_junctions_tab = np.unique(intersecting_junctions_tab)
    #     hash_tag_ignored = np.unique(hash_tag_ignored)
    #     junction_vector = np.add(dest_junction, -start_junction)
    #     for hash_tag in intersecting_junctions_tab:
    #         if hash_tag in hash_tag_ignored:
    #             continue
    #         for waypoint in self.world.waypoints_dic[str(hash_tag)]:
    #             negative_test_vector = np.add(
    #                 carla_vector2array_2d(waypoint.transform.location).round(decimals=2),
    #                 - start_junction
    #             )
    #             alpha1 = np.abs(np.cross(negative_test_vector, junction_vector))
    #             negative_test_vector += np.array([
    #                 np.cos(np.deg2rad(waypoint.transform.rotation.yaw)),
    #                 np.sin(np.deg2rad(waypoint.transform.rotation.yaw))
    #             ]).round(decimals=2)
    #             alpha2 = np.abs(np.cross(negative_test_vector, junction_vector))
    #             if alpha1 > alpha2:
    #                 waypoints_to_giveaway.append(waypoint)
    #     self.waypoints = waypoints_to_giveaway

    def set_waypoint_to_giveaway(self, waypoints_to_giveaway):
        self.waypoints = waypoints_to_giveaway

    def get_dangerous_waypoints(self):
        dangerous_waypoints = []
        for waypoint in self.waypoints:
            waypoint_location = waypoint.transform.location
            dists2checkpoints = []
            check_points_pixels = []
            for height in self.heights_check_points:
                check_point = carla.Location(
                    x=waypoint_location.x, y=waypoint_location.y, z=waypoint_location.z + height
                )
                dists2checkpoints.append(dist_location_2d(
                    check_point,
                    self.world.occupancy_mapper.egoVehicle.get_location()
                ))
                check_points_pixels.append(
                    world2pixel([[carla_vector2array_3d(check_point)]], self.world.recorder.last_rec)[0][0]
                )
            dists2checkpoints = np.array(dists2checkpoints)
            check_points_pixels = np.array(check_points_pixels)
            if (
                    dists2checkpoints < 2 * self.world.occupancy_mapper.visual_horizon +
                    self.world.occlusion_tolerance_detection
            ).all():
                if (dists2checkpoints < 2 * self.world.occupancy_mapper.visual_horizon).all():
                    to_close = False
                    for tracker in self.world.occupancy_mapper.tracker_list:
                        to_close = False
                        if tracker.visible and dist(
                                carla_vector2array_2d(waypoint_location).round(decimals=2),
                                tracker.pos
                        ) <= 1.7 * tracker.one_circle_radius:
                            to_close = True
                            break
                    if to_close:
                        continue
                    check_points_pixels_occluded = True
                    for pixel in check_points_pixels:
                        if pixel_in_image(pixel, self.world.recorder.last_rec) and not point_is_occluded(
                                pixel, self.world.recorder.last_rec,
                                tolerance=self.world.occlusion_tolerance_detection):
                            check_points_pixels_occluded = False
                            break
                    if pixel_in_image(check_points_pixels[0], self.world.recorder.last_rec) and \
                            check_points_pixels_occluded:
                        dangerous_waypoints.append(waypoint)
                elif pixel_in_image(check_points_pixels[0], self.world.recorder.last_rec):
                    dangerous_waypoints.append(waypoint)
        return dangerous_waypoints

    def get_critical_waypoints(self, dangerous_waypoints):
        lane_dic = {}
        critical_waypoints = []
        for waypoint in dangerous_waypoints:
            dic_to_hash = {
                "road_id": waypoint.road_id,
                "section_id": waypoint.section_id,
                "lane_id": waypoint.lane_id
            }
            s = json.dumps(dic_to_hash, sort_keys=True)
            waypoint_id = str(int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10 ** 8)
            if waypoint_id not in lane_dic:
                lane_dic[waypoint_id] = [waypoint]
            else:
                lane_dic[waypoint_id].append(waypoint)
        vehicle_loc = carla_vector2array_2d(self.world.vehicle.get_location()).round(decimals=2)
        for index in lane_dic:
            critical_waypoint = None
            min_distance_to_ego = float('inf')
            for waypoint in lane_dic[index]:
                waypoint_loc = carla_vector2array_2d(waypoint.transform.location).round(decimals=2)
                distance_to_ego = dist(waypoint_loc, vehicle_loc)
                if distance_to_ego < min_distance_to_ego:
                    min_distance_to_ego = distance_to_ego
                    critical_waypoint = waypoint
            critical_waypoints.append(critical_waypoint)
        return critical_waypoints
