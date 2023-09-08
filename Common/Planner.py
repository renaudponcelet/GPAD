import csv
import json
import math

import carla
import matplotlib.pyplot as plt
import numpy as np
import pygame
from ..Approaches.MMRIS.MMRISLauncher import RISLauncher
from ..Approaches.SGSPA.SGSPALauncher import VISLauncher
from ..Common.Utils.ControlLauncher import Controller
from ..Common.Utils.carla_utils import carla_vector2array_2d, array_wp2nd_array
from ..Common.Utils.controller import VehiclePIDController
from ..Common.Utils.utils import get_lazy_path, get_index_offset, norm_x_y, \
    find_nearest_vector, PathToShort
from ..Common.Utils.OccupancyViewer import OccupancyViewer
from ..Approaches.Common.occupancy_mapper import Tracker
from ..Approaches.Common.occupancy_mapper.VirtualVehicleGenerator import VirtualVehicleGenerator
from ..Common.Utils.agents.navigation.global_route_planner import GlobalRoutePlanner


class Planner(object):
    def __init__(self, world, client, start_in_autopilot, fps):
        self.world = world
        self.rec_step = 0
        self.counter = 0
        self.blocked = False
        self.kool_down = False
        self.time_since_kool_down = 0
        self.time_since_light_is_red = None
        self.ego_blocked_time = 10 if not self.world.scenario_mode else self.world.scenario_data["ego_blocked_time"]
        self.world.scenario_data["ego_blocked_time"] = self.ego_blocked_time
        self.kool_down_time = 10 if not self.world.scenario_mode else self.world.scenario_data["kool_down_time"]
        self.world.scenario_data["kool_down_time"] = self.kool_down_time
        self.auto_mode_first_time = None
        self.ris_launcher = self.vis_launcher = None
        if 'ris-path' in self.world.planner_list:
            self.ris_launcher = RISLauncher(self)
        if 'vis-speed' in self.world.planner_list:
            self.vis_launcher = VISLauncher(self)
            if self.ris_launcher is None:
                self.ris_launcher = RISLauncher(self)
        #  I set a global plan thank to carla waypoints (uses .xodr file)
        self.global_path = self.get_global_path()
        self.path_list = []
        self.global_paths_alternative = self.get_global_paths_alternative()
        self.global_path_point = array_wp2nd_array(self.global_path, route_mode=True).round(decimals=2)
        self.local_lane = None
        self.waypoints_to_giveaway = self.set_waypoint_to_giveaway()
        if world.awareness == 'vv':
            self.virtual_vehicle_generator = VirtualVehicleGenerator(world)
            self.virtual_vehicle_generator.set_waypoint_to_giveaway(self.waypoints_to_giveaway)
            self.world.occupancy_mapper.set_virtual_vehicle_generator(self.virtual_vehicle_generator)
        self.client = client
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self.prev_local_goal = None
        world.vehicle.set_autopilot(self._autopilot_enabled)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
        # Here I set the controller
        # It's based on carla's controller, but I modified lateral pid controller in a stanley controller.
        # For the lateral controller, K_P is the only used parameter
        # To tune the PID I made an identification
        self.selected_plan = None
        dt = 1.0 / fps
        args_lateral = {
            'K_P': 0.9,
            'K_D': 0.0,
            'K_I': 0.0} if not self.world.scenario_mode else self.world.scenario_data["pid_args_lateral"]
        self.world.scenario_data["pid_args_lateral"] = args_lateral
        args_lateral["dt"] = dt
        args_longitudinal = {
            'K_P': 6,
            'K_D': 0.1,
            'K_I': 0.4} if not self.world.scenario_mode else self.world.scenario_data["pid_args_longitudinal"]
        self.world.scenario_data["pid_args_longitudinal"] = args_longitudinal
        args_longitudinal["dt"] = dt
        # PID tuned for 20 fps
        self.PID_controller = VehiclePIDController(
            world.vehicle,
            args_lateral=args_lateral,
            args_longitudinal=args_longitudinal,
            max_throttle=1.0,
            max_brake=0.3,
            max_steering=0.5
        )
        self.L = 2.7 if not self.world.scenario_mode else self.world.scenario_data["L"]
        self.world.scenario_data["L"] = self.L
        self.L_control = 2.7 if not self.world.scenario_mode else self.world.scenario_data["L_control"]
        self.world.scenario_data["L_control"] = self.L_control
        self.path = None
        self.speed_plan = None
        self.rec = {
            "first_time": None,
            "step": [],
            "location": [],
            "rotation": [],
            "velocity": [],
            "speed": [],
            "angular_velocity": [],
            "acceleration": [],
            "acceleration_norm": [],
            "throttle": [],
            "steer": [],
            "brake": [],
            "target_speed": [],
            "time_in_junction": 0.0,
            "global_path": None
        }
        self.controller = Controller(self, dt)
        self.display_last_thread = None
        self.rec_last_thread = None
        self.last_out_index = None
        self.time_since_ego_stop = None
        self.tracker_list = []
        for actor in world.actors_list:
            self.tracker_list.append(Tracker(world, actor, planner=self))
        for actor in world.additional_actor:
            self.tracker_list.append(Tracker(world, actor, stopped_car=True))
        world.occupancy_mapper.set_trackers(self.tracker_list)
        world.save_rec_scenario()

    def parse_events(self, clock):
        self.world.update_time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == pygame.K_BACKSPACE:
                    self.world.restart()
                elif event.key == pygame.K_F1:
                    self.world.hud.toggle_info()
                elif event.key == pygame.K_TAB:
                    self.world.camera_manager.toggle_camera()
                elif event.key == pygame.K_c and pygame.key.get_mods() & pygame.KMOD_SHIFT:
                    self.world.next_weather(reverse=True)
                elif event.key == pygame.K_c:
                    self.world.next_weather()
                elif event.key == pygame.K_BACKQUOTE:
                    self.world.camera_manager.next_sensor()
                elif pygame.K_0 < event.key <= pygame.K_9:
                    self.world.camera_manager.set_sensor(event.key - 1 - pygame.K_0)()
                elif event.key == pygame.K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    self.world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        # I shunted carla autopilot with the following code for ego-vehicle
        if not self._autopilot_enabled or self.world.respawned:
            if self.world.respawned:
                self._autopilot_enabled = False
                self.last_out_index = None
                self.time_since_ego_stop = None
                self.auto_mode_first_time = None
                self.prev_local_goal = None
                self.rec_step = 0
                self.path = None
                self.speed_plan = None
                self.rec = {
                    "first_time": None,
                    "step": [],
                    "location": [],
                    "rotation": [],
                    "velocity": [],
                    "speed": [],
                    "angular_velocity": [],
                    "acceleration": [],
                    "acceleration_norm": [],
                    "throttle": [],
                    "steer": [],
                    "brake": [],
                    "target_speed": [],
                    "time_in_junction": 0.0,
                    "global_path": None
                }
                if 'ris-path' in self.world.planner_list:
                    self.ris_launcher.respawn()
                if 'vis-speed' in self.world.planner_list:
                    self.vis_launcher.respawn()
                self.world.reset_respawn()
                self.world.collision_sensor.reset_history()
            if self.world.auto_mode:
                if self.world.auto_mode_delay != 0.0:
                    if self.auto_mode_first_time is None:
                        self.auto_mode_first_time = 0
                    else:
                        if self.world.world_time - self.auto_mode_first_time >= self.world.auto_mode_delay:
                            self._autopilot_enabled = True
                else:
                    self._autopilot_enabled = True
            # Update here world
            self.world.visual_horizon = self.world.vehicle_speed_limit * (self.world.time_horizon +
                                                                          self.world.time_gap)
            self.world.nb_step = int(self.world.time_horizon // self.world.time_step_res)
            self.world.ego_offset = 2 * self.world.vehicle.bounding_box.extent.x / self.world.nb_circles
            self.world.ego_radius = (math.sqrt((self.world.ego_offset / 2) ** 2
                                               + self.world.vehicle.bounding_box.extent.y ** 2))
            self.world.recorder.get_frame()
            if self.world.display:
                print("Step ", self.world.occupancy_viewer_ris.step)
            # Here i re-set some var in case autopilot is enabled again after a first use
            # I also apply manual control to debug
            if 'ris-path' in self.world.planner_list:
                self.ris_launcher.time_since_replan = 0
            if 'vis-speed' in self.world.planner_list:
                self.vis_launcher.time_since_replan = 0
            self._parse_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
            self.world.vehicle.apply_control(self._control)
        else:
            self.world.recorder.get_frame()
            if self.world.display:
                print("Step ", self.world.occupancy_viewer_ris.step)
            if 'ris-path' in self.world.planner_list:
                self.ris_launcher.time_since_replan = self.world.world_time - self.ris_launcher.time_at_last_replan
            if 'vis-speed' in self.world.planner_list:
                self.vis_launcher.time_since_replan = self.world.world_time - self.vis_launcher.time_at_last_replan
            # vehicle_location_nd = carla_vector2array_2d(vehicle_location)
            # Here i get current lane : it can be an alternative lane to global_path, global_path, or None
            #                           when no information is given from map.
            if self.path is None:
                local_goal = self.get_local_lane()
            else:
                local_goal = self.get_local_lane(carla.Location(x=self.path[-1][0], y=self.path[-1][1]))
            if local_goal is None:
                raise Exception("local_goal is None")
            # Here i get local goal from global goal and i assure it is on the road
            # If ego get to the last global goal it closes the program and print "success"
            # local_goal is now called in get_local_lane
            if isinstance(local_goal, str):
                if "vis-path" in self.world.planner_list and self.world.display:
                    self.vis_launcher.display_tread.join()
                if self.world.display and self.world.rec:
                    result_viewer = OccupancyViewer('results')
                    # result_viewer.add_line(self.rec["target_speed"], color='#0000ff')
                    filter_speed = None
                    self.rec['acceleration_norm'][0] = [self.rec['acceleration_norm'][0][0], 0.0]
                    for i, _ in enumerate(self.rec["speed"]):
                        if i == 0 or i == len(self.rec['speed']):
                            continue
                        if filter_speed is None:
                            filter_speed = np.mean(self.rec["speed"][i - 1:i + 1], axis=0)
                        else:
                            filter_speed = np.vstack((filter_speed, np.mean(self.rec["speed"][i - 2:i + 2], axis=0)))
                    result_viewer.add_line(np.array(filter_speed).round(decimals=3), color='#ff0000', label='speed')
                    filter_acc = None
                    for i, _ in enumerate(self.rec["acceleration_norm"]):
                        if i == 0 or i == 1 or i == 2 or i == len(self.rec['acceleration_norm']) \
                                or i == len(self.rec['acceleration_norm']) - 1:
                            continue
                        if filter_acc is None:
                            filter_acc = np.mean(self.rec["acceleration_norm"][i - 3:i + 2], axis=0)
                        else:
                            filter_acc = np.vstack(
                                (filter_acc, np.mean(self.rec["acceleration_norm"][i - 3:i + 2], axis=0)))
                    result_viewer.add_line(np.array(filter_acc).round(decimals=2), color='#0000ff',
                                           label='acceleration')
                    result_viewer.show_figure(
                        title='results',
                        rec_folder=self.world.rec_folder
                    )
                print(local_goal)
                return True
            if len(self.world.collision_sensor.get_collision_history()) != 0:
                with open("_out/results.csv", 'a') as outFile:
                    fileWriter = csv.writer(outFile)
                    fileWriter.writerow([0])
                print("collision")
                return True
            if "ris-path" in self.world.planner_list:
                # You can launch several thread with different speed profile here
                self.ris_launcher.launch(
                    local_goal=local_goal,
                    path=self.path
                )
                if self.world.planning_mode is 'unique':
                    self.ris_launcher.call(goal=local_goal)

            if 'vis-speed' in self.world.planner_list:
                if self.world.planning_mode is 'unique':
                    self.path = get_lazy_path(
                        pos=self.world.vehicle.get_location(),
                        global_path_point=self.global_path_point,
                        world=self.world
                    )
                    self.vis_launcher.launch(path=self.path, local_goal=local_goal)
                    self.vis_launcher.call()
                else:
                    if self.path is None:
                        self.ris_launcher.call(
                            goal=local_goal
                        )
                        self.selected_plan = 'ris-path'
                    else:
                        indexes = find_nearest_vector(self.path,
                                                      carla_vector2array_2d(self.world.vehicle.get_location()))
                        self.path = self.path[min(indexes):]
                        if len(self.path) > 3:
                            self.vis_launcher.launch(path=self.path, local_goal=local_goal)
                            try:
                                self.vis_launcher.call()
                                vis_score = self.vis_launcher.get_score()
                            except PathToShort:
                                vis_score = -1
                        else:
                            vis_score = -1
                        if vis_score == 1:
                            self.selected_plan = 'vis-speed'
                        else:
                            self.ris_launcher.call(goal=local_goal)
                            ris_score = self.ris_launcher.get_score()
                            if ris_score >= vis_score:
                                self.selected_plan = 'ris-path'
                            else:
                                self.selected_plan = 'vis-speed'
            if self.world.planning_mode is 'unique':
                if 'ris-path' in self.world.planner_list:
                    self.path, self.speed_plan = self.ris_launcher.get_results()
                elif 'vis-speed' in self.world.planner_list:
                    self.path, self.speed_plan = self.vis_launcher.get_results()
                    if self.counter % (self.world.replan_ratio + 1) == 0:
                        self.ris_launcher.path = self.path
                        self.ris_launcher.speed_plan = None
                        self.ris_launcher.call(display_only=True)
                else:
                    raise Exception("planner not in planner list")
            else:
                print(self.selected_plan)
                if self.selected_plan is 'ris-path':
                    self.path, self.speed_plan = self.ris_launcher.get_results()
                elif self.selected_plan is 'vis-speed':
                    self.path, self.speed_plan = self.vis_launcher.get_results()
                    if self.counter % (self.world.replan_ratio + 1) == 0:
                        self.ris_launcher.path = self.path
                        self.ris_launcher.speed_plan = None
                        self.ris_launcher.call(display_only=True)
                else:
                    raise Exception("planner not in planner list")

            # We get control from the motion plan here
            if self.speed_plan is not None and isinstance(self.speed_plan, str) \
                    and self.speed_plan == 'in poly':
                self.speed_plan = None
                with open(self.world.rec_folder + "/recorder/collision_detector.csv", 'a') as outFile:
                    fileWriter = csv.writer(outFile)
                    fileWriter.writerow([self.world.world_time])
            self.controller.update(self.path, self.speed_plan)
            self.controller.run()
            if np.linalg.norm(carla_vector2array_2d(self.world.vehicle.get_velocity()).round(decimals=2)) * 3.6 < 2:
                if self.time_since_ego_stop is None:
                    self.time_since_ego_stop = self.world.world_time
                    self.blocked = False
                else:
                    if str(self.world.occupancy_mapper.traffic_light_state) == 'Red':
                        self.kool_down = True
                        self.time_since_kool_down = self.world.world_time
                        if self.time_since_light_is_red is None:
                            self.time_since_light_is_red = self.world.world_time
                        self.blocked = False
                    else:
                        if self.time_since_light_is_red is not None:
                            self.time_since_ego_stop += self.world.world_time - self.time_since_light_is_red
                            self.time_since_light_is_red = None
                        if self.world.world_time - self.time_since_ego_stop > self.ego_blocked_time:
                            self.blocked = True
                        else:
                            self.blocked = False

                    if self.world.world_time - self.time_since_ego_stop > 10 * self.ego_blocked_time:
                        with open("_out/results.csv", 'a') as outFile:
                            fileWriter = csv.writer(outFile)
                            fileWriter.writerow([2])
                        print("Ego is blocked")
                        return True
            else:
                self.time_since_ego_stop = None
                self.blocked = False
            if self.world.rec:
                self.get_rec()
        self.counter += 1

    def _parse_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[pygame.K_UP] or keys[pygame.K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self._steer_cache -= steer_increment
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[pygame.K_DOWN] or keys[pygame.K_s] else 0.0
        self._control.hand_brake = keys[pygame.K_SPACE]

    def get_global_paths_alternative(self, debug_display=False):
        global_paths_alternative = {}
        for step in self.global_path:
            step_waypoint = step[0]
            if step_waypoint.is_junction:
                step_hash = step_waypoint.junction_id
                lane_type = "junction"
            else:
                step_hash = hash((str(step_waypoint.road_id), str(step_waypoint.section_id)))
                lane_type = "driving"
            if step_hash not in global_paths_alternative:
                self.path_list.append(step_hash)
                lane_hash = hash(
                    (str(step_waypoint.road_id), str(step_waypoint.section_id), str(step_waypoint.lane_id))
                )
                global_paths_alternative[step_hash] = {
                    "type": lane_type,
                    "alternative_lane": {
                        lane_hash: {
                            "status": 's',
                            "lane": step_waypoint.next_until_lane_end(self.world.global_path_interval)
                        }
                    }
                }
                if lane_type == "junction":
                    global_paths_alternative[step_hash]["junction"] = step_waypoint.get_junction().get_waypoints(
                        carla.LaneType.Driving)
                    global_paths_alternative[step_hash]["alternative_lane"] = {}
                else:
                    if str(step_waypoint.lane_change) == "Left" or str(step_waypoint.lane_change) == "Both" or \
                            str(step_waypoint.left_lane_marking.type) == "Broken":
                        if str(step_waypoint.lane_change) == "Left":
                            global_paths_alternative[step_hash]["alternative_lane"][lane_hash]["status"] = 'r'
                        left_lane = step_waypoint.get_left_lane()
                        left_status = ''
                        while left_lane is not None:
                            if str(step_waypoint.lane_change) == "Left" or str(step_waypoint.lane_change) == "Both":
                                left_status += 'l'
                                lane = left_lane.next_until_lane_end(self.world.global_path_interval)
                            else:
                                left_status = 'o'
                                lane = left_lane.previous_until_lane_start(self.world.global_path_interval)[:-1]
                            left_hash = hash(
                                (str(left_lane.road_id), str(left_lane.section_id), str(left_lane.lane_id))
                            )
                            global_paths_alternative[step_hash]["alternative_lane"][left_hash] = {
                                "status": left_status,
                                "lane": lane
                            }
                            if str(left_lane.lane_change) == "Left" and (
                                    str(step_waypoint.lane_change) == "Left"
                                    or str(step_waypoint.lane_change) == "Both"):
                                left_lane = left_lane.get_left_lane()
                            else:
                                left_lane = None
                    if str(step_waypoint.lane_change) == "Right" or str(step_waypoint.lane_change) == "Both":
                        if str(step_waypoint.lane_change) == "Right":
                            global_paths_alternative[step_hash]["alternative_lane"][lane_hash]["status"] = 'l'
                        right_lane = step_waypoint.get_right_lane()
                        right_status = ''
                        while right_lane is not None:
                            right_status += 'r'
                            right_hash = hash(
                                (str(right_lane.road_id), str(right_lane.section_id), str(right_lane.lane_id))
                            )
                            global_paths_alternative[step_hash]["alternative_lane"][right_hash] = {
                                "status": right_status,
                                "lane": right_lane.next_until_lane_end(self.world.global_path_interval)
                            }
                            if str(right_lane.lane_change) == "Right":
                                right_lane = right_lane.get_right_lane()
                            else:
                                right_lane = None
            else:
                continue
        for i, hash_tag in enumerate(self.path_list):
            # handle the special case od junction
            if global_paths_alternative[hash_tag]["type"] == "junction":
                if i < 1 or i >= len(self.path_list) - 1:
                    if i == 0:
                        raise Exception('The plan start in a junction')
                    else:
                        raise Exception('The plan end in a junction')
                else:
                    pre_selection = []
                    for junction_id, junction_section in enumerate(global_paths_alternative[hash_tag]["junction"]):
                        previous_hash = self.path_list[i - 1]
                        if global_paths_alternative[previous_hash]["type"] == "junction":
                            raise Exception("it can't support successive junction yet")
                        wp_start = junction_section[0]
                        wp_start_hash = hash(
                            (str(wp_start.road_id), str(wp_start.section_id), str(wp_start.lane_id))
                        )
                        wp_start_reverse = junction_section[1]
                        wp_start_reverse_hash = hash(
                            (str(wp_start_reverse.road_id), str(wp_start_reverse.section_id), str(wp_start_reverse.lane_id))
                        )
                        # check if junction start links with an alternative lane end
                        for alt_hash in global_paths_alternative[previous_hash]["alternative_lane"]:
                            next_wps = (
                                global_paths_alternative[previous_hash]["alternative_lane"][alt_hash]["lane"][-1]
                            ).next(self.world.global_path_interval)
                            next_wps_reverse = (
                                global_paths_alternative[previous_hash]["alternative_lane"][alt_hash]["lane"][-1]
                            ).previous(self.world.global_path_interval)
                            for next_wp in next_wps + next_wps_reverse:
                                next_wp_hash = hash(
                                    (str(next_wp.road_id), str(next_wp.section_id), str(next_wp.lane_id))
                                )
                                if next_wp_hash == wp_start_hash:
                                    pre_selection.append([
                                        junction_section[0],
                                        junction_section[1],
                                        global_paths_alternative[previous_hash]["alternative_lane"][alt_hash]["status"],
                                        junction_id
                                    ])
                                elif next_wp_hash == wp_start_reverse_hash:
                                    pre_selection.append([
                                        junction_section[0],
                                        junction_section[1],
                                        global_paths_alternative[previous_hash]["alternative_lane"][alt_hash]["status"],
                                        junction_id
                                    ])
                    for junction_section in pre_selection:
                        wp_end = junction_section[1]
                        wps_end_next = wp_end.next(self.world.global_path_interval)
                        wp_end_reverse = junction_section[0]
                        wp_end_reverse_next = wp_end_reverse.previous(self.world.global_path_interval)
                        for wp_end_next in wps_end_next + wp_end_reverse_next:
                            wp_end_hash = hash(
                                (str(wp_end_next.road_id), str(wp_end_next.section_id), str(wp_end_next.lane_id))
                            )
                            next_hash = self.path_list[i + 1]
                            if global_paths_alternative[next_hash]["type"] == "junction":
                                raise Exception('The plan start or end in a junction')
                            for alt_hash in global_paths_alternative[next_hash]["alternative_lane"]:
                                next_lane_wp = global_paths_alternative[next_hash][
                                    "alternative_lane"][alt_hash]["lane"][0]
                                previous_wp_hash = hash(
                                    (str(next_lane_wp.road_id), str(next_lane_wp.section_id), str(next_lane_wp.lane_id))
                                )
                                if previous_wp_hash == wp_end_hash:
                                    wp_start = junction_section[0]
                                    junction_hash = hash(
                                        (str(wp_start.road_id), str(wp_start.section_id), str(wp_start.lane_id)))
                                    global_paths_alternative[hash_tag]["alternative_lane"][junction_hash] = {
                                        "status": junction_section[2] + global_paths_alternative[next_hash][
                                            "alternative_lane"][alt_hash]["status"],
                                        "lane": wp_start.next_until_lane_end(self.world.global_path_interval),
                                        "junction_id": junction_section[3]
                                    }
        if debug_display:
            for alternative in global_paths_alternative:
                for lane in global_paths_alternative[alternative]["alternative_lane"]:
                    if global_paths_alternative[alternative]['type'] == 'driving':
                        if global_paths_alternative[alternative]["alternative_lane"][lane]["status"] == 'l':
                            color = 'b'
                        elif global_paths_alternative[alternative]["alternative_lane"][lane]["status"] == 'r':
                            color = 'r'
                        elif global_paths_alternative[alternative]["alternative_lane"][lane]["status"] == 's':
                            color = 'g'
                        else:
                            color = 'y'
                    elif global_paths_alternative[alternative]['type'] == 'junction':
                        color = 'c'
                    else:
                        raise Exception("Not a known type")
                    lane_array = array_wp2nd_array(
                        global_paths_alternative[alternative]["alternative_lane"][lane]["lane"]).round(decimals=2)
                    plt.plot(lane_array[:, 0], lane_array[:, 1], color)
            plt.show()
        return global_paths_alternative

    def get_lane_from_location(self, location):
        waypoint = self.world.map.get_waypoint(location)
        if waypoint.is_junction:
            global_waypoint_hash = waypoint.junction_id
        else:
            global_waypoint_hash = hash((str(waypoint.road_id), str(waypoint.section_id)))
        waypoint_hash = hash((str(waypoint.road_id), str(waypoint.section_id), str(waypoint.lane_id)))
        lane = self.get_lane_from_hash(global_waypoint_hash, waypoint_hash)
        if lane is None and not waypoint.is_junction:
            new_waypoints = waypoint.next(self.world.global_path_interval)
            for waypoint in new_waypoints:
                global_waypoint_hash = hash((str(waypoint.road_id), str(waypoint.section_id)))
                waypoint_hash = hash((str(waypoint.road_id), str(waypoint.section_id), str(waypoint.lane_id)))
                lane = self.get_lane_from_hash(global_waypoint_hash, waypoint_hash)
        elif lane is None and waypoint.is_junction:
            min_dist = float('inf')
            for junction_hash in self.global_paths_alternative[global_waypoint_hash]["alternative_lane"]:
                lane_array = array_wp2nd_array(self.global_paths_alternative[global_waypoint_hash]["alternative_lane"][
                                                   junction_hash]["lane"]).round(decimals=2)
                ego_location_array = carla_vector2array_2d(location).round(decimals=2)
                index = find_nearest_vector(lane_array, ego_location_array)[1]
                dist = np.linalg.norm(np.add(lane_array[index], - ego_location_array))
                if dist < min_dist:
                    min_dist = dist
                    lane = self.get_lane_from_hash(global_waypoint_hash, junction_hash)
        return lane

    def get_lane_from_hash(self, global_waypoint_hash, waypoint_hash):
        if global_waypoint_hash in self.global_paths_alternative:
            if waypoint_hash in self.global_paths_alternative[global_waypoint_hash]["alternative_lane"]:
                lane = self.global_paths_alternative[global_waypoint_hash]["alternative_lane"][waypoint_hash]
            else:
                lane = None
        else:
            lane = None
        return lane

    def get_local_lane(self, end_last_path_location=None):
        if self.local_lane is None:
            if end_last_path_location is None:
                self.local_lane = [self.get_lane_from_location(self.world.vehicle.get_location())]
            else:
                self.local_lane = []
                ego_location_waypoint = self.world.map.get_waypoint(self.world.vehicle.get_location())
                if ego_location_waypoint.is_junction:
                    ego_location_global_hash = ego_location_waypoint.junction_id
                else:
                    ego_location_global_hash = hash((str(ego_location_waypoint.road_id),
                                                     str(ego_location_waypoint.section_id)))
                temp_local_lane = self.get_lane_from_location(end_last_path_location)
                start_local_lane_waypoint = temp_local_lane["lane"][0]
                start_local_lane_status = temp_local_lane["status"][0]
                if start_local_lane_waypoint.is_junction:
                    temp_local_lane_global_hash = start_local_lane_waypoint.junction_id
                else:
                    temp_local_lane_global_hash = hash((str(start_local_lane_waypoint.road_id),
                                                        str(start_local_lane_waypoint.section_id)))
                if temp_local_lane_global_hash != ego_location_global_hash:
                    hash_before_temp_lane = [ego_location_global_hash]
                    flag = True
                    for hash_tag in self.path_list:
                        if flag and ego_location_global_hash != hash_tag:
                            pass
                        elif flag and ego_location_global_hash == hash_tag:
                            flag = False
                        elif not flag and temp_local_lane_global_hash != hash_tag:
                            hash_before_temp_lane.append(hash_tag)
                        else:
                            break
                    for hash_tag in hash_before_temp_lane:
                        for alt_hash in self.global_paths_alternative[hash_tag]["alternative_lane"]:
                            intermediate_lane = self.get_lane_from_hash(hash_tag, alt_hash)
                            if intermediate_lane["status"][-1] == start_local_lane_status:
                                self.local_lane += [intermediate_lane]
                self.local_lane += [temp_local_lane]
        else:
            previous_lane = self.local_lane
            if end_last_path_location is None:
                new_lane = [self.get_lane_from_location(self.world.vehicle.get_location())]
            else:
                new_lane = []
                ego_location_waypoint = self.world.map.get_waypoint(self.world.vehicle.get_location())
                if ego_location_waypoint.is_junction:
                    ego_location_global_hash = ego_location_waypoint.junction_id
                else:
                    ego_location_global_hash = hash((str(ego_location_waypoint.road_id),
                                                     str(ego_location_waypoint.section_id)))
                temp_local_lane = self.get_lane_from_location(end_last_path_location)
                start_local_lane_waypoint = temp_local_lane["lane"][0]
                start_local_lane_status = temp_local_lane["status"][0]
                if start_local_lane_waypoint.is_junction:
                    temp_local_lane_global_hash = start_local_lane_waypoint.junction_id
                else:
                    temp_local_lane_global_hash = hash((str(start_local_lane_waypoint.road_id),
                                                        str(start_local_lane_waypoint.section_id)))
                if temp_local_lane_global_hash != ego_location_global_hash:
                    hash_before_temp_lane = [ego_location_global_hash]
                    flag = True
                    for hash_tag in self.path_list:
                        if flag and ego_location_global_hash != hash_tag:
                            pass
                        elif flag and ego_location_global_hash == hash_tag:
                            flag = False
                        elif not flag and temp_local_lane_global_hash != hash_tag:
                            hash_before_temp_lane.append(hash_tag)
                        else:
                            break
                    for hash_tag in hash_before_temp_lane:
                        for alt_hash in self.global_paths_alternative[hash_tag]["alternative_lane"]:
                            intermediate_lane = self.get_lane_from_hash(hash_tag, alt_hash)
                            if intermediate_lane["status"][-1] == start_local_lane_status:
                                new_lane += [intermediate_lane]
                new_lane += [temp_local_lane]
            if new_lane is not None:
                self.local_lane = new_lane
            else:
                print("Warning : lane update is temporary unusable")
                self.local_lane = [previous_lane[0]]
            if previous_lane[0]["status"][-1] != self.local_lane[0]["status"][0]:
                print("Lane changed")
                self.kool_down = True
                self.time_since_kool_down = self.world.world_time
            else:
                if self.world.world_time - self.time_since_kool_down >= self.kool_down_time:
                    self.kool_down = False
        local_goal = self.get_local_goal()
        if isinstance(local_goal, str):
            return local_goal
        goal_waypoint = self.world.map.get_waypoint(carla.Location(local_goal[0], local_goal[1]))
        if goal_waypoint.is_junction:
            goal_global_hash = goal_waypoint.junction_id
        else:
            goal_global_hash = hash((str(goal_waypoint.road_id), str(goal_waypoint.section_id)))
        if self.local_lane[0]["lane"][0].is_junction:
            local_global_hash = self.local_lane[0]["lane"][0].junction_id
        else:
            local_global_hash = hash(
                (str(self.local_lane[0]["lane"][0].road_id), str(self.local_lane[0]["lane"][0].section_id)))
        flag = False
        for hash_tag in self.path_list:
            if hash_tag != local_global_hash and not flag:
                continue
            elif hash_tag == local_global_hash and not flag:
                if hash_tag == goal_global_hash:
                    return local_goal
                flag = True
                continue
            else:
                for alt_hash in self.global_paths_alternative[hash_tag]["alternative_lane"]:
                    intermediate_lane = self.get_lane_from_hash(hash_tag, alt_hash)
                    if self.local_lane[-1]["status"][-1] == intermediate_lane["status"][0] or (
                            self.local_lane[-1]["status"][-1] is 'o' and intermediate_lane["status"][0] is 's'):
                        self.local_lane += [intermediate_lane]
                        break
                if hash_tag == goal_global_hash:
                    return local_goal
        raise Exception("Goal is not on global path")

    def get_local_goal(self):
        if np.linalg.norm(
                carla_vector2array_2d(self.world.vehicle.get_location()
                                      ) - self.global_path_point[-1]) < self.world.goal_tolerance:
            with open("_out/results.csv", 'a') as outFile:
                fileWriter = csv.writer(outFile)
                fileWriter.writerow([1])
            return "success"
        index, index_offset = get_index_offset(
            self.global_path_point, carla_vector2array_2d(self.world.vehicle.get_location()), self.world)
        out_index = index + index_offset
        if self.last_out_index is not None:
            if out_index < self.last_out_index:
                out_index = self.last_out_index + 1
        self.last_out_index = out_index
        if out_index >= len(self.global_path_point):
            return self.global_path_point[-1]
        return self.global_path_point[out_index]

    def get_rec(self):
        current_time = self.world.world_time
        if self.rec["first_time"] is None:
            self.rec["first_time"] = [current_time]
            self.rec["step"].append(0.0)
            step_time = 0.0
        else:
            step_time = current_time - self.rec["first_time"][0]
            self.rec["step"].append(step_time)
        self.rec["global_path"] = self.global_path_point.tolist()
        location = self.world.vehicle.get_location()
        self.rec["location"].append([step_time, location.x, location.y, location.z])
        # rotation = vehicle.get_transform().rotation
        # self.rec["rotation"].append([step_time, rotation.pitch, rotation.yaw, rotation.roll])
        velocity = self.world.vehicle.get_velocity()
        # self.rec["velocity"].append([velocity.x, velocity.y, velocity.z])
        self.rec["speed"].append([step_time, norm_x_y(velocity)])
        # angular_velocity = vehicle.get_angular_velocity()
        # self.rec["angular_velocity"].append([angular_velocity.x, angular_velocity.y, angular_velocity.z])
        acceleration = self.world.vehicle.get_acceleration()
        # self.rec["acceleration"].append([acceleration.x, acceleration.y, acceleration.z])
        self.rec["acceleration_norm"].append([step_time, norm_x_y(acceleration)])
        self.rec["time_in_junction"] = (self.world.time_in_junction['last_time']
                                        - self.world.time_in_junction['first_time'])
        # control = vehicle.get_control()
        # self.rec["throttle"].append(control.throttle)
        # self.rec["steer"].append(control.steer)
        # self.rec["brake"].append(control.brake)
        # self.rec["target_speed"].append([step_time, target_speed[1] if target_speed is not None else 0.0])
        with open(self.world.rec_folder + '/scenario_file.json', 'w') as f:
            json.dump(self.world.scenario_data, f, indent=4)
        with open(self.world.rec_folder + '/rec_file.json', 'w') as f:
            json.dump(self.rec, f, indent=4)

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == pygame.K_ESCAPE) or (key == pygame.K_q and pygame.key.get_mods() & pygame.KMOD_CTRL)

    def get_global_path(self):
        global_path = []
        global_route_planner = GlobalRoutePlanner(self.world.map, self.world.global_path_interval)
        # global_route_planner.setup()
        last_destination = None
        for destination in self.world.destinations:
            if last_destination is None:
                start = self.world.spawn_waypoint.transform.location
            else:
                start = last_destination.transform.location
            global_path += global_route_planner.trace_route(start, destination.transform.location)
            last_destination = destination
        return np.array(global_path)

    def set_waypoint_to_giveaway(self):
        root_waypoint_to_giveaway = []
        for step in self.global_paths_alternative:
            if self.global_paths_alternative[step]["type"] == 'junction':
                junctions_id_in_global_path = []
                junctions_in_global_path = []
                for junction in self.global_paths_alternative[step]['alternative_lane']:
                    junction_id = self.global_paths_alternative[step]['alternative_lane'][junction]["junction_id"]
                    junctions_id_in_global_path.append(junction_id)
                    junctions_in_global_path.append(self.global_paths_alternative[step]['junction'][junction_id])
                for junction_id, junction in enumerate(self.global_paths_alternative[step]['junction']):
                    if junction_id in junctions_id_in_global_path:
                        continue
                    junction_array = array_wp2nd_array(junction)
                    for junction_in_global_path in junctions_in_global_path:
                        junction_in_global_path_array = array_wp2nd_array(junction_in_global_path)
                        v = np.cross(np.add(junction_array, - junction_in_global_path_array[0]),
                                     junction_in_global_path_array
                                     )
                        if v[0] < 0 <= v[1]:
                            root_waypoint_to_giveaway.append(junction[0])
                            break
        waypoint_to_giveaway = None
        for root_waypoint in root_waypoint_to_giveaway:
            if waypoint_to_giveaway is None:
                waypoint_to_giveaway = np.array([root_waypoint])
            else:
                waypoint_to_giveaway = np.concatenate((waypoint_to_giveaway, [root_waypoint]))
            previous_waypoints = root_waypoint.previous(self.world.global_path_interval)
            waypoint_to_giveaway = np.concatenate((waypoint_to_giveaway, previous_waypoints))
            for previous_waypoint in previous_waypoints:
                lane = previous_waypoint.previous_until_lane_start(self.world.global_path_interval)[:-1]
                waypoint_to_giveaway = np.concatenate((waypoint_to_giveaway, lane))
        if waypoint_to_giveaway is None:
            waypoint_to_giveaway = np.array([])
        return waypoint_to_giveaway
