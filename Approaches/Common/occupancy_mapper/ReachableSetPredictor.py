import carla
import numpy as np
from ....Common.Utils.carla_utils import carla_vector2array_2d


def get_reachable_waypoints(world, pos, speed, yaw=None, virtual_mode=False, previous_waypoints=None, behind=False):
    norm_speed = np.linalg.norm(speed)
    if norm_speed < 0.1 * world.vehicle_speed_limit:
        previous_waypoints = None
    step_dist = norm_speed * world.time_step_res
    acc_dist = 0
    waypoints = []
    waypoints_point = []
    temp_waypoints = []
    temp_waypoints_points = []
    for t in range(int((world.time_horizon + world.time_gap) // world.time_step_res)):
        if t == 0:
            if previous_waypoints is None:
                waypoint = world.map.get_waypoint(carla.Location(x=pos[0], y=pos[1]))
                if yaw is None:
                    yaw = np.deg2rad(waypoint.transform.rotation.yaw)
            else:
                previous_waypoints_concat = np.concatenate(np.array(previous_waypoints), axis=0)
                dist = float('inf')
                near_waypoint = None
                for waypoint in previous_waypoints_concat:
                    waypoint_point = np.array(
                        [waypoint.transform.location.x, waypoint.transform.location.y]).round(decimals=2)
                    temp_dist = np.linalg.norm(np.add(waypoint_point, -np.array([pos[0], pos[1]]).round(decimals=2)))
                    if temp_dist < dist:
                        dist = temp_dist
                        near_waypoint = waypoint
                if dist > 1:
                    waypoint = world.map.get_waypoint(carla.Location(x=pos[0], y=pos[1]))
                else:
                    waypoint = near_waypoint
                if yaw is None:
                    yaw = np.deg2rad(waypoint.transform.rotation.yaw)
            waypoints.append([waypoint])
            waypoints_point.append(np.array([np.array([
                pos[0],
                pos[1],
                yaw
            ])]))
        else:
            acc_dist += step_dist
            if acc_dist >= world.global_path_interval:
                count = 0
                while acc_dist >= world.global_path_interval:
                    count += 1
                    acc_dist -= world.global_path_interval
                for waypoint in waypoints[-1]:
                    temp_waypoints += list(waypoint.next(count * world.global_path_interval))
                    # check for available right driving lanes
                    if (str(waypoint.lane_change) == "Right" or str(waypoint.lane_change) == "Both") and not behind:
                        right_w = waypoint.get_right_lane()
                        if right_w and str(right_w.lane_type) == "Driving":
                            temp_waypoints += list(right_w.next(count * world.global_path_interval))
                for waypoint in temp_waypoints:
                    loc = carla_vector2array_2d(waypoint.transform.location).round(decimals=2)
                    temp_waypoints_points.append(np.array([loc[0],
                                                           loc[1],
                                                           np.deg2rad(waypoint.transform.rotation.yaw)]))
                if virtual_mode:
                    temp_waypoints_points = np.concatenate((temp_waypoints_points, waypoints_point[-1]), axis=0)

                waypoints_point.append(np.unique(temp_waypoints_points, axis=0))
                waypoints.append(temp_waypoints)
                temp_waypoints = []
                temp_waypoints_points = []
            else:
                waypoints_point.append(waypoints_point[-1])
                waypoints.append(waypoints[-1])
    return waypoints_point, waypoints


class ReachableSetPredictor:
    name = "dummy predictor : all possibilities are considered at constant speed"

    def __init__(self):
        self.last_yaw = None
        self.buffer_time = 0.5
        self.speed_buffer = []

    def get_estimation(self, world, pos, speed, waypoints, heading, behind=False, priority=True):
        last_speed = carla_vector2array_2d(speed[-1]["speed"]).round(decimals=2)
        current_time = speed[-1]["timestamp"]
        self.speed_buffer.append(np.array([current_time, last_speed], dtype=object))
        if not priority:
            self.speed_buffer.append(np.array([current_time, 0], dtype=object))
        mean_speed = self.update_buffer(current_time)
        future_pos_estimation, waypoints_out = get_reachable_waypoints(
            world, pos, mean_speed, yaw=heading, previous_waypoints=waypoints, behind=behind
        )

        return future_pos_estimation, waypoints_out, mean_speed

    def update_buffer(self, current_time):
        i = 0
        for speed in self.speed_buffer:
            if current_time - speed[0] > self.buffer_time:
                i += 1
        self.speed_buffer = self.speed_buffer[i:]
        return np.mean(self.speed_buffer, axis=0)[1]
