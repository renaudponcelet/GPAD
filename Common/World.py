import json
import math
import os
import sys

import carla
import numpy as np

from ..Approaches.MMRIS.MMRIS import MMRIS
from ..Approaches.SGSPA.SGSPA import SGSPA
from ..Common.Utils.CameraManager import CameraManager
from ..Common.Utils.CollisionSensor import CollisionSensor
from ..Common.Utils.carla_utils import carla_vector2array_2d, get_nearest_tr, find_weather_presets, get_actor_display_name
from ..Common.Utils.utils import norm_x_y, super_dic
from ..Common.Utils.OccupancyViewer import OccupancyViewer
from ..Approaches.Common.occupancy_mapper import OccupancyMapper
from ..Approaches.Common.occupancy_mapper import Recorder


class World(object):
    def __init__(self, carla_world, client, hud, clock, cl_args, fps):
        self.world_time = 0
        self.planner = None
        self.vehicle_physics_off = []
        self.time_in_junction = {
            'first_time': 0,
            'last_time': 0
        }
        self.allow_threading = cl_args.allow_threading
        self.auto_mode = cl_args.auto_mode
        self.auto_mode_delay = cl_args.delay
        self.planner_list = cl_args.planner_list
        self.planning_mode = 'unique' if len(self.planner_list) == 1 else "parallel"
        self.display = cl_args.display
        self.rec = cl_args.rec
        self.scenario_mode = cl_args.scenario_mode
        self.rec_scenario = cl_args.rec_scenario
        self.rec_folder = None
        self.scenario_path = 'Scenarios/'
        self.scenario_data = {}
        if self.scenario_mode:
            self.scenario_name = cl_args.scenario_name
            super_scenario_data = None
            if '/' in self.scenario_name:
                index = self.scenario_name.find('/')
                super_scenario_file_path = self.scenario_path + self.scenario_name[:index] + '/data.json'
                try:
                    with open(super_scenario_file_path) as f:
                        super_scenario_data = json.load(f)
                except FileNotFoundError:
                    super_scenario_data = None
            self.scenario_file_path = self.scenario_path + self.scenario_name + '/data.json'
            with open(self.scenario_file_path) as f:
                self.scenario_data = json.load(f)
            if super_scenario_data is not None:
                self.scenario_data = super_dic(super_scenario_data, self.scenario_data)
        if self.rec_scenario:
            self.rec_file_path = self.scenario_path + 'temp/data.json'
        self.clock = clock
        self.awareness = cl_args.awareness if not self.scenario_mode else self.scenario_data["awareness"]
        self.scenario_data["awareness"] = self.awareness
        self.time_gap = 2 if not self.scenario_mode else self.scenario_data["time_gap"]
        self.scenario_data["time_gap"] = self.time_gap
        self.margin = 0.1 if not self.scenario_mode else self.scenario_data["margin"]
        self.scenario_data["margin"] = self.margin
        self.tolerance = 0.1 if not self.scenario_mode else self.scenario_data["tolerance"]
        self.scenario_data["tolerance"] = self.tolerance
        self.nb_paths = 4 if not self.scenario_mode else self.scenario_data["nb_paths"]
        self.scenario_data["nb_paths"] = self.nb_paths
        self.cross_security = cl_args.cross_sec
        self.scenario_data["cross_security"] = self.cross_security
        self.apf_params = [1, 1, 1] if not self.scenario_mode else self.scenario_data["apf_params"]
        self.scenario_data["apf_params"] = self.apf_params
        self.fps = fps
        self.occlusion_tolerance_detection = 2.0 if not self.scenario_mode else \
            self.scenario_data["occlusion_tolerance_detection"]
        self.scenario_data["occlusion_tolerance_detection"] = self.occlusion_tolerance_detection
        self.nb_actors = 20 if not self.scenario_mode else self.scenario_data["nb_actors"]
        self.client = client
        self.seed = np.random.randint(0, 999) if \
            not self.scenario_mode or self.scenario_data["seed"] == 'None' else self.scenario_data["seed"]
        self.scenario_data["seed"] = self.seed
        np.random.seed(self.seed)
        self.tm = None
        if self.nb_actors > 0:
            self.tm = client.get_trafficmanager()
            self.tm.set_random_device_seed(self.seed)
        if self.rec_scenario:
            self.scenario_data["nb_actors"] = self.nb_actors
            self.scenario_data["map"] = cl_args.map
        self.world = carla_world
        self.hud = hud
        self.vehicle = None
        self.__vehicle_speed_limit = 8.6
        self.__vehicle_speed = 0.0
        self.__vehicle_acc = 0.0
        self.occupancy_mapper = None
        self.global_path_interval = 1 if not self.scenario_mode else self.scenario_data["global_path_interval"]
        self.scenario_data["global_path_interval"] = self.global_path_interval
        self.time_horizon = 4 if not self.scenario_mode else self.scenario_data["time_horizon"]  # m
        self.scenario_data["time_horizon"] = self.time_horizon
        self.__visual_horizon = 8.6 * (self.time_horizon + self.time_gap)
        self.nb_step = None
        self.collision_sensor = None
        self.goal_tolerance = 10

        # ego_vehicle_info
        self.nb_circles = 3 if not self.scenario_mode else self.scenario_data["nb_circles"]
        self.scenario_data["nb_circles"] = self.nb_circles
        self.random_agent = True if not self.scenario_mode else self.scenario_data["random_agent"]
        self.scenario_data["random_agent"] = self.random_agent
        self.ego_offset = None
        self.ego_radius = None
        self.max_acc = 2.5 if not self.scenario_mode else self.scenario_data["max_acc"]
        self.scenario_data["max_acc"] = self.max_acc
        self.max_acc_emergency = 4.0 if not self.scenario_mode else self.scenario_data["max_acc_emergency"]
        self.scenario_data["max_acc_emergency"] = self.max_acc_emergency

        self.max_jerk = 10000

        self.destinations = []
        self.camera_manager = None
        self.actors_list = None
        self.additional_actor = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        with open('GPAD/vehicle_info.json', 'r') as f:
            self.vehicles_info = json.load(f)[0]
        self.replan_ratio = 4 if not self.scenario_mode else self.scenario_data["replan_ratio"]
        self.scenario_data["replan_ratio"] = self.replan_ratio
        self.time_step_res = 1 / fps
        self.RIS = None
        self.VIS = None
        self.recorder = None
        self.occupancy_viewer_ris = None
        self.occupancy_viewer_vis = None
        self.waypoints = None
        self.map = None
        self.respawned = False
        self.spawn_point = None
        self.spawn_waypoint = None
        self.intersection_waypoints = None
        self.topology = None
        self.ego_init_transform = None
        self.waypoints_dic = {}
        self.actors_init_transform_list = []
        # self.time_step = int(1000 // fps) / 1000  # s
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def set_planner(self, planner):
        self.planner = planner

    def restart(self, delay=None):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.no_rendering_mode = True
        settings.fixed_delta_seconds = 1.0 / self.fps
        if self.tm is not None:
            self.tm.set_synchronous_mode(True)
        self.world.apply_settings(settings)
        if delay is not None:
            self.auto_mode_delay = delay
        self.map = self.world.get_map()
        self.waypoints = self.map.generate_waypoints(self.global_path_interval / 2)
        for waypoint in self.waypoints:
            hash_tag = hash((str(waypoint.road_id), str(waypoint.section_id), str(waypoint.lane_id)))
            if str(hash_tag) not in self.waypoints_dic:
                self.waypoints_dic[str(hash_tag)] = [waypoint]
            else:
                self.waypoints_dic[str(hash_tag)].append(waypoint)
        self.topology = self.map.get_topology()
        self.intersection_waypoints = []
        for waypoint in self.waypoints:
            if waypoint.is_intersection:
                self.intersection_waypoints.append(waypoint)
        used_spawn_points = []
        # Keep the same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random vehicle blueprint.
        type_id = None
        if not self.scenario_mode or self.rec_scenario:
            list_type_id = []
            for vehicle_id in self.vehicles_info["specific"]:
                list_type_id.append(vehicle_id)
            if self.rec_scenario:
                self.scenario_data['list_type_id'] = list_type_id
        else:
            list_type_id = self.scenario_data['list_type_id']
        blueprint = None
        while type_id not in list_type_id:
            blueprint = np.random.choice(self.world.get_blueprint_library().filter('vehicle'))
            type_id = blueprint.id
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', '224, 0, 0')
        # Spawn the vehicle.
        spawn_points = self.map.get_spawn_points()
        if not self.scenario_mode or self.rec_scenario:
            while self.vehicle is None:
                self.spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
                used_spawn_points.append(self.spawn_point)
                self.spawn_waypoint = self.map.get_waypoint(self.spawn_point.location)
                self.destinations = [np.random.choice(self.spawn_waypoint.next(500))]
                self.vehicle = self.world.try_spawn_actor(blueprint, self.spawn_point)
            if self.rec_scenario:
                self.scenario_data['spawn_point'] = [
                    self.spawn_point.location.x, self.spawn_point.location.y, self.spawn_point.location.z]
                self.scenario_data['destinations'] = [[
                    self.destinations[0].transform.location.x,
                    self.destinations[0].transform.location.y,
                    self.destinations[0].transform.location.z
                ]]
        else:
            spawn_point_target = carla.Location(
                x=self.scenario_data['spawn_point'][0],
                y=self.scenario_data['spawn_point'][1],
                z=self.scenario_data['spawn_point'][2]
            )
            self.spawn_point = self.map.get_waypoint(spawn_point_target).transform
            self.spawn_waypoint = self.map.get_waypoint(self.spawn_point.location)
            for dest in self.scenario_data['destinations']:
                self.destinations.append(self.map.get_waypoint(carla.Location(
                    x=dest[0],
                    y=dest[1],
                    z=dest[2]
                )))
            self.vehicle = self.world.try_spawn_actor(blueprint, self.spawn_point)
            if self.vehicle is None:
                self.spawn_point = get_nearest_tr(spawn_points, self.spawn_point.location)
                self.vehicle = self.world.try_spawn_actor(
                    blueprint, self.spawn_point)
                if self.vehicle is None:
                    raise Exception("vehicle is None")
            self.spawn_waypoint = self.map.get_waypoint(self.spawn_point.location)
        used_spawn_points.append(self.spawn_point)
        self.ego_init_transform = self.vehicle.get_transform()
        self.nb_step = int(self.time_horizon // self.time_step_res)
        self.ego_offset = 2 * self.vehicle.bounding_box.extent.x / self.nb_circles
        self.ego_radius = math.sqrt((self.ego_offset / 2) ** 2 + self.vehicle.bounding_box.extent.y ** 2)

        # Set up the sensors.
        if self.display:
            c = 0
            while True:
                if "rec_" + str(c) not in os.listdir("_out/"):
                    os.makedirs("_out/rec_" + str(c))
                    break
                else:
                    c += 1
            self.rec_folder = "_out/rec_" + str(c)
            if self.rec and "recorder" not in os.listdir("_out"):
                os.mkdir(self.rec_folder + "/recorder")
            if "metadata" not in os.listdir(self.rec_folder):
                os.mkdir(self.rec_folder + "/metadata")
            if "viewer" not in os.listdir(self.rec_folder):
                os.mkdir(self.rec_folder + "/viewer")
            self.occupancy_viewer_ris = OccupancyViewer('viewer')
            if 'vis-speed' in self.planner_list:
                if "speed_plan_viewer" not in os.listdir(self.rec_folder):
                    os.mkdir(self.rec_folder + "/speed_plan_viewer")
                self.occupancy_viewer_vis = OccupancyViewer('speed_plan_viewer')
            if self.rec:
                os.mkdir(self.rec_folder + '/results')
            out = open(self.rec_folder + '/out.log', 'w')
            sys.stdout = out
            sys.stderr = out
        self.occupancy_mapper = OccupancyMapper(self)
        middle_map_scenario = carla_vector2array_2d(self.spawn_point.location).round(decimals=2)
        counter = 1
        for destination in self.destinations:
            counter += 1
            middle_map_scenario = np.add(middle_map_scenario, carla_vector2array_2d(
                destination.transform.location).round(decimals=2))
        middle_map_scenario /= counter
        self.camera_manager = CameraManager(self.vehicle, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)

        if self.planning_mode == 'unique':
            if 'ris-path' in self.planner_list:
                self.recorder = Recorder(self)
                self.RIS = MMRIS(world=self)
            elif 'vis-speed' in self.planner_list:
                self.recorder = Recorder(self)
                self.VIS = SGSPA(world=self)
            else:
                raise Exception('Wrong planner name')
        elif self.planning_mode is 'parallel':
            if 'ris-path' in self.planner_list:
                self.recorder = Recorder(self)
                self.RIS = MMRIS(world=self)
            if 'vis-speed' in self.planner_list:
                if self.recorder is None:
                    self.recorder = Recorder(self)
                self.VIS = SGSPA(world=self)
        else:
            raise Exception('This planning mode does not exist')

        actor_type = get_actor_display_name(self.vehicle)
        self.hud.notification(actor_type)
        # Set up MMRIS
        # # Spawn actors
        self.actors_list = []
        self.additional_actor = []
        actor_dic = None
        if self.rec_scenario:
            actor_dic = {}
        if not self.scenario_mode or self.rec_scenario:
            for x in range(0, self.nb_actors):
                if self.rec_scenario:
                    actor_dic[str(x)] = {}
                spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
                while spawn_point in used_spawn_points:
                    spawn_point = np.random.choice(spawn_points) if spawn_points else carla.Transform()
                if self.rec_scenario:
                    actor_dic[str(x)]['spawn_point'] = [spawn_point.location.x,
                                                        spawn_point.location.y,
                                                        spawn_point.location.z]
                type_id = None
                while type_id not in list_type_id:
                    blueprint = np.random.choice(self.world.get_blueprint_library().filter('vehicle'))
                    if blueprint.has_attribute('color'):
                        blueprint.set_attribute('color', '0, 0, 254')
                    type_id = blueprint.id
                npc = self.world.try_spawn_actor(blueprint, spawn_point)
                if npc is not None:
                    used_spawn_points.append(spawn_point)
                    self.actors_list.append(npc)
                    npc.set_autopilot()
                    print('created %s' % npc.type_id)
                if self.rec_scenario:
                    self.scenario_data['actors'] = actor_dic
        else:
            for x in range(0, self.nb_actors):
                type_id = None
                while type_id not in list_type_id:
                    blueprint = np.random.choice(self.world.get_blueprint_library().filter('vehicle'))
                    if blueprint.has_attribute('color'):
                        blueprint.set_attribute('color', '0, 0, 254')
                    type_id = blueprint.id
                if str(x) in self.scenario_data["actors"]:
                    spawn_point_target = carla.Location(
                        x=self.scenario_data["actors"][str(x)]["spawn_point"][0],
                        y=self.scenario_data["actors"][str(x)]["spawn_point"][1],
                        z=self.scenario_data["actors"][str(x)]["spawn_point"][2])
                else:
                    spawn_point_target = np.random.choice(spawn_points).location if spawn_points else carla.Location()
                    self.scenario_data["actors"][str(x)] = {}
                    self.scenario_data["actors"][str(x)]["spawn_point"] = [
                        spawn_point_target.x, spawn_point_target.y, spawn_point_target.z
                    ]
                spawn_point = self.map.get_waypoint(spawn_point_target).transform
                npc = self.world.try_spawn_actor(blueprint, spawn_point)
                if npc is None:
                    spawn_point = get_nearest_tr(spawn_points, spawn_point.location)
                    while spawn_point in used_spawn_points:
                        spawn_point = np.random.choice(spawn_points)
                    npc = self.world.try_spawn_actor(
                        blueprint, spawn_point)
                    if npc is None:
                        print("a vehicle failed to spawn")
                if npc is not None:
                    used_spawn_points.append(spawn_point)
                    self.actors_list.append(npc)
                    self.actors_init_transform_list.append(npc.get_transform())

                    npc.set_autopilot()
                    if "ignore_lights_percentage" in self.scenario_data["actors"][str(x)]:
                        self.tm.ignore_lights_percentage(
                            npc, self.scenario_data["actors"][str(x)]["ignore_lights_percentage"]
                        )
                    if "percentage_speed_difference" in self.scenario_data["actors"][str(x)]:
                        self.tm.vehicle_percentage_speed_difference(
                            npc, self.scenario_data["actors"][str(x)]["percentage_speed_difference"])
                    print('created %s' % npc.type_id)
        if "stop_car" not in self.scenario_data:
            self.scenario_data["stop_car"] = False
        if self.scenario_data["stop_car"]:
            blueprint = np.random.choice(self.world.get_blueprint_library().filter('vehicle.volkswagen.t2'))
            if blueprint.has_attribute('color'):
                blueprint.set_attribute('color', '0, 0, 0')
            for pos in self.scenario_data["stop_car_pos"]:
                stop_car = None
                while stop_car is None:
                    stop_car = self.world.try_spawn_actor(
                        blueprint,
                        np.random.choice(spawn_points)
                    )
                self.additional_actor.append(stop_car)
                stop_car.set_simulate_physics(False)
                stop_car.set_transform(
                    carla.Transform(
                        carla.Location(x=pos[0], y=pos[1], z=2),
                        carla.Rotation(yaw=pos[2])
                    )
                )
                self.vehicle_physics_off.append(stop_car)
                control = carla.VehicleControl(hand_brake=True)
                stop_car.apply_control(control)
        self.world.tick()
        for vehicle in self.vehicle_physics_off:
            vehicle.set_simulate_physics(True)
        physics_control = self.vehicle.get_physics_control()
        physics_control.damping_rate_zero_throttle_clutch_engaged = 1.0
        self.vehicle.apply_physics_control(physics_control)
        self.time_in_junction["first_time"] = 0
        self.time_in_junction["last_time"] = 0

    @property
    def visual_horizon(self):
        return self.__visual_horizon

    @visual_horizon.setter
    def visual_horizon(self, distance):
        if distance < 1:
            # self.__visual_horizon = 2 * 8.6 * (self.time_horizon + self.time_gap)
            self.__visual_horizon = 8.6 * (self.time_horizon + self.time_gap)
        else:
            self.__visual_horizon = distance

    @property
    def vehicle_speed_limit(self):
        self.vehicle_speed_limit = self.vehicle.get_speed_limit() / 3.6
        return self.__vehicle_speed_limit

    @vehicle_speed_limit.setter
    def vehicle_speed_limit(self, speed_limit):
        if speed_limit < 1:
            self.__vehicle_speed_limit = 8.6
        else:
            self.__vehicle_speed_limit = speed_limit

    @property
    def vehicle_speed(self):
        self.vehicle_speed = norm_x_y(self.vehicle.get_velocity())
        return self.__vehicle_speed

    @vehicle_speed.setter
    def vehicle_speed(self, speed):
        self.__vehicle_speed = speed

    @property
    def vehicle_acc(self):
        self.vehicle_acc = norm_x_y(self.vehicle.get_acceleration())
        return self.__vehicle_acc

    @vehicle_acc.setter
    def vehicle_acc(self, acc):
        self.__vehicle_acc = acc

    def save_rec_scenario(self):
        if self.rec_scenario:
            with open(self.rec_file_path, 'w') as json_file:
                json.dump(self.scenario_data, json_file)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
                     self.camera_manager.sensor,
                     self.collision_sensor.sensor,
                     self.vehicle] + self.actors_list + self.additional_actor
        if self.tm is not None:
            self.tm.set_synchronous_mode(False)
        self.recorder.destroy()
        for actor in actors:
            if actor is not None:
                actor.destroy()

    def respawn(self, delay):
        self.recorder.destroy()
        self.vehicle.set_simulate_physics(False)
        for actor in self.actors_list:
            actor.set_simulate_physics(False)
        safe_location = carla.Location(
            x=self.ego_init_transform.location.x,
            y=self.ego_init_transform.location.y,
            z=self.ego_init_transform.location.z + 2
        )
        safe_transform = carla.Transform(
            location=safe_location,
            rotation=self.ego_init_transform.rotation
        )
        self.vehicle.set_transform(safe_transform)
        for i, actor in enumerate(self.actors_list):
            safe_location = carla.Location(
                x=self.actors_init_transform_list[i].location.x,
                y=self.actors_init_transform_list[i].location.y,
                z=self.actors_init_transform_list[i].location.z + 2
            )
            safe_transform = carla.Transform(
                location=safe_location,
                rotation=self.ego_init_transform.rotation
            )
            actor.set_transform(safe_transform)
        self.vehicle.set_simulate_physics(True)
        for actor in self.actors_list:
            actor.set_simulate_physics(True)
        self.auto_mode_delay = delay
        if self.display:
            self.occupancy_viewer_ris.restart()
            c = 0
            while True:
                if 'rec_' + str(c) not in os.listdir('_out/'):
                    os.makedirs("_out/rec_" + str(c))
                    break
                else:
                    c += 1
            self.rec_folder = "_out/rec_" + str(c)
            if self.rec and "recorder" not in os.listdir("_out"):
                os.mkdir(self.rec_folder + "/recorder")
            if "metadata" not in os.listdir(self.rec_folder):
                os.mkdir(self.rec_folder + "/metadata")
            if "viewer" not in os.listdir(self.rec_folder):
                os.mkdir(self.rec_folder + "/viewer")
            if 'vis-speed' in self.planner_list:
                self.occupancy_viewer_vis.restart()
                if "speed_plan_viewer" not in os.listdir(self.rec_folder):
                    os.mkdir(self.rec_folder + "/speed_plan_viewer")
        if 'ris' in self.planner_list:
            self.recorder.set_sensor(self, 'depth', 'ris')
        else:
            self.recorder.set_sensor(self, 'depth', 'vis')
        self.respawned = True

    def reset_respawn(self):
        self.respawned = False

    def update_time(self):
        self.world_time += 1.0 / self.fps
