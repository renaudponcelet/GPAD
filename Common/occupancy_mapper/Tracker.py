import math

import numpy as np
from Common.Utils.carla_utils import carla_vector2array_3d, carla_vector2array_2d, pixel_in_image, \
    point_is_occluded
from Common.Utils.carla_utils import world2pixel, get_bounding_box_shape_circles, array_wp2nd_array
from Common.Utils.utils import dist
from Common.occupancy_mapper.ReachableSetPredictor import ReachableSetPredictor


class Tracker:
    name = "class to track dynamic participants (cars) from sensors"

    def __init__(self, world, vehicle, planner=None, stopped_car=False):
        if planner is not None:
            self.waypoints_to_giveaway = planner.waypoints_to_giveaway
        else:
            self.waypoints_to_giveaway = []
        self.stopped_car = stopped_car
        self.stopped = stopped_car
        self.world = world
        self.vehicle = vehicle
        self.one_circle_radius = math.sqrt(
            self.vehicle.bounding_box.extent.x ** 2 + self.vehicle.bounding_box.extent.y ** 2
        )
        self.pos = None
        self.visible = None
        self.time_gap = world.time_gap
        self.behind = False
        self.priority = True
        self.future_pos_estimation = None
        self.waypoints = None
        self.speed = []
        self.mean_speed = None
        self.vehicle_shape_circles, self.vehicle_shape_radius = get_bounding_box_shape_circles(
            self.vehicle.bounding_box
        )
        self.predictor = ReachableSetPredictor()

    def update(self):
        self.pos = carla_vector2array_3d(self.vehicle.get_location()).round(decimals=2)
        ego_pos = carla_vector2array_2d(self.world.vehicle.get_location()).round(decimals=2)
        self.update_rec()
        if self.world.awareness == "omniscient":
            self.visible = True
        elif dist(self.pos[:2], ego_pos) <= 2 * self.world.visual_horizon:
            pixel = world2pixel([[self.pos]], self.world.recorder.current_rec)[0][0]
            # TODO : to be realistic we have to check occlusion with at least 2 cameras, on on the front and one the
            #  back of the car
            if pixel[2] >= 0:
                in_image = True if pixel_in_image(pixel, self.world.recorder.current_rec) else False
                if in_image:
                    self.visible = not self.check_occlusion()
        else:
            self.visible = False
        ego_heading = np.deg2rad(self.world.vehicle.get_transform().rotation.yaw)
        if ego_heading <= 0:
            ego_heading += 2 * math.pi
        elif ego_heading > 2 * math.pi:
            ego_heading -= 2 * math.pi
        ego_heading_vector = np.array([math.cos(ego_heading), math.sin(ego_heading)]).round(decimals=2)
        diff_pos_vector = np.add(self.pos[:2], - ego_pos)
        if np.dot(ego_heading_vector, diff_pos_vector) < (
                self.world.vehicle.bounding_box.extent.x + self.vehicle.bounding_box.extent.x):
            self.behind = True
        else:
            self.behind = False
        if self.visible:
            if self.stopped_car:
                self.future_pos_estimation = []
                for t in range(int((self.world.time_horizon + self.world.time_gap) // self.world.time_step_res)):
                    self.future_pos_estimation.append(np.array([np.array([
                        self.pos[0],
                        self.pos[1],
                        np.deg2rad(self.vehicle.get_transform().rotation.yaw)
                    ])]))
                self.future_pos_estimation = np.array(self.future_pos_estimation)
            else:
                if self.waypoints_to_giveaway.shape[0] > 0 and (self.waypoints is None or np.min(np.linalg.norm(
                        array_wp2nd_array(self.waypoints_to_giveaway) - np.array(
                            [self.waypoints[0][0].transform.location.x,
                             self.waypoints[0][0].transform.location.y]
                        ), axis=1)) < 5):
                    self.priority = True
                else:
                    self.priority = False
                self.future_pos_estimation, self.waypoints, self.mean_speed = self.predictor.get_estimation(
                    self.world,
                    self.pos[:2],
                    self.speed,
                    self.waypoints,
                    np.deg2rad(self.vehicle.get_transform().rotation.yaw),
                    behind=self.behind,
                    priority=self.priority
                )
                if np.linalg.norm(self.mean_speed) <= 0.5:
                    self.stopped = True
                else:
                    self.stopped = False
                if self.behind:
                    flag = False
                    for t, future_pos_slice in enumerate(self.future_pos_estimation):
                        time = t * self.world.time_step_res
                        if time > self.world.time_gap:
                            break
                        for pos in future_pos_slice:
                            diff_pos_vector = np.add(pos[:2], - ego_pos)
                            dot_product = np.dot(ego_heading_vector, diff_pos_vector)
                            if dot_product >= - (
                                    self.world.vehicle.bounding_box.extent.x + self.vehicle.bounding_box.extent.x) and \
                                    np.linalg.norm(diff_pos_vector) ** 2 - dot_product ** 2 < (
                                    self.world.vehicle.bounding_box.extent.y + self.vehicle.bounding_box.extent.y) ** 2:
                                self.time_gap = (t - 1) * self.world.time_step_res
                            if dot_product >= - (
                                    self.world.vehicle.bounding_box.extent.x + self.vehicle.bounding_box.extent.x):
                                print("dot_product is positive again")
                                flag = True
                                break
                        if flag:
                            break
            if self.world.display:
                count = 0
                for future_pos_slice in self.future_pos_estimation:
                    if count == 0:
                        for pos in future_pos_slice:
                            color = '#0000ff' if self.behind else '#00ffff'
                            label = '\n(behind)' if self.behind else ''
                            if self.priority:
                                color = '#c000ff'
                                label = '\n(priority)'
                            self.world.occupancy_viewer_ris.add_circle(
                                np.array([pos[0], pos[1], 0.2]).round(decimals=2),
                                color=color,
                                frame=self.world.recorder.current_frame,
                                screens=[0, 1, 2, 3],
                                label='occupancy \n prevision' + label
                            )
                        count += 1
                    elif count == 5:
                        count = 0
                    else:
                        count += 1
        else:
            if self.world.display:
                self.world.occupancy_viewer_ris.add_circle([self.pos[0], self.pos[1], 1.5], color='#ffff00',
                                                           frame=self.world.recorder.current_frame,
                                                           screens=[0, 1, 2, 3],
                                                           label='hidden vehicles'
                                                           )
            self.future_pos_estimation = None

    def check_occlusion(self):
        # bounding_box = self.vehicle.bounding_box
        # loc = bounding_box.location
        vehicle_loc = self.vehicle.get_location()
        # ext = bounding_box.extent
        # check_points = [
        #     [loc.x + vehicle_loc.x + ext.x, loc.y + vehicle_loc.y + ext.y, loc.z + vehicle_loc.z],
        #     [loc.x + vehicle_loc.x + ext.x, loc.y + vehicle_loc.y - ext.y, loc.z + vehicle_loc.z],
        #     [loc.x + vehicle_loc.x - ext.x, loc.y + vehicle_loc.y - ext.y, loc.z + vehicle_loc.z],
        #     [loc.x + vehicle_loc.x - ext.x, loc.y + vehicle_loc.y + ext.y, loc.z + vehicle_loc.z],
        #     [loc.x + vehicle_loc.x + ext.x, loc.y + vehicle_loc.y + ext.y, loc.z + vehicle_loc.z + ext.z],
        #     [loc.x + vehicle_loc.x + ext.x, loc.y + vehicle_loc.y - ext.y, loc.z + vehicle_loc.z + ext.z],
        #     [loc.x + vehicle_loc.x - ext.x, loc.y + vehicle_loc.y - ext.y, loc.z + vehicle_loc.z + ext.z],
        #     [loc.x + vehicle_loc.x - ext.x, loc.y + vehicle_loc.y + ext.y, loc.z + vehicle_loc.z + ext.z]
        # ]
        check_points = [
            [vehicle_loc.x, vehicle_loc.y, vehicle_loc.z]
        ]
        check_pixels = world2pixel([check_points], self.world.recorder.current_rec)[0]
        is_occluded = True
        for pixel in check_pixels:
            if pixel_in_image(pixel, self.world.recorder.current_rec) and not point_is_occluded(
                    pixel, self.world.recorder.current_rec, tolerance=self.world.occlusion_tolerance_detection):
                is_occluded = False
                return is_occluded
        return is_occluded

    def update_rec(self):
        measurement = {
            "speed": self.vehicle.get_velocity(),
            "timestamp": self.world.world_time,
        }
        self.speed.append(measurement)
