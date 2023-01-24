import math
import threading

import numpy as np
from PythonAPI.GPAD.Common.Utils.carla_utils import get_static_path, carla_vector2array_2d, get_index_offset
from PythonAPI.carla.agents.tools.misc import get_speed


class GetSpeed(threading.Thread):
    def __init__(self, world, vis, path, frame, display):
        super(GetSpeed, self).__init__()
        self.frame = frame
        self.vis = vis
        self.path_ref = path
        self.path = None
        self.speed_plan = None
        self.display = display
        self.world = world

    def run(self):
        self.speed_plan, self.path = self.vis.get_speed(
            path=self.path_ref
        )


class VISLauncher:
    name = "launcher for VIS thread"

    def __init__(self, planner):
        self.planner = planner
        self.path = None
        self.speed_plan = None
        self.quality_flag = True
        self.ego_location = None
        self.step = 0
        self.vis_thread = None
        self.time_since_replan = 0
        self.time_at_last_replan = 0
        self.display_tread = None
        self.stopped_at_a_red_light = False
        self.static_path = None

    def launch(self, path, local_goal):
        if self.planner.counter % (self.planner.world.replan_ratio + 1) == 0:
            self.ego_location = carla_vector2array_2d(
                self.planner.world.vehicle.get_transform().location).round(decimals=2)
            self.time_since_replan = 0
            self.planner.world.occupancy_mapper.update_obstacles()
            self.stopped_at_a_red_light = False
            if (str(self.planner.world.occupancy_mapper.traffic_light_state) == 'Red'
                    and get_speed(self.planner.world.vehicle) < 2):
                print('ego is stopped at a red light')
                self.stopped_at_a_red_light = True
            ego_heading = np.deg2rad(self.planner.world.vehicle.get_transform().rotation.yaw)
            self.renew_path(path, local_goal, ego_heading)
            if self.planner.world.allow_threading and not self.stopped_at_a_red_light:
                self.vis_thread = GetSpeed(
                    world=self.planner.world,
                    vis=self.planner.world.VIS,
                    path=self.static_path,
                    frame=self.planner.world.recorder.current_frame,
                    display=self.planner.world.display
                )
                self.vis_thread.start()

    def call(self):
        if self.planner.counter % (self.planner.world.replan_ratio + 1) == 0:
            if self.stopped_at_a_red_light:
                (self.path, self.speed_plan) = (self.static_path[:, :2], None)
            else:
                if self.planner.world.allow_threading:
                    self.vis_thread.join(timeout=6)
                    (self.path, self.speed_plan) = (self.vis_thread.path, self.vis_thread.speed_plan)
                else:
                    (self.speed_plan, self.path) = self.planner.world.VIS.get_speed(path=self.static_path[:, :2])
            if self.planner.world.display:
                if self.planner.rec["first_time"] is None:
                    self.planner.rec["first_time"] = [0]
            self.draw()
            self.time_at_last_replan = self.planner.world.world_time
            self.time_since_replan = 0
        elif self.planner.world.planning_mode is "parallel" and self.planner.selected_plan != "vis-speed":
            self.speed_plan = None

    def draw(self):
        self.planner.world.occupancy_viewer_vis.show_figure(
            title='SGSPA approach',
            frame=self.planner.world.recorder.current_frame,
            ref_time=self.planner.rec["first_time"][0],
            rec_folder=self.planner.world.rec_folder,
            world=self.planner.world,
            x_lim=[0, self.planner.world.time_horizon],
            x_label='time (second)',
            y_lim=[0, self.planner.world.visual_horizon],
            y_label='s (meters)',
            time=self.time_at_last_replan,
            aspect="auto"
        )

    def respawn(self):
        self.vis_thread = None
        self.path = None
        self.speed_plan = None

    def get_results(self):
        return self.path, self.speed_plan

    def get_score(self, f0=4, f1=2):
        # if we are on 'l' or 'o' lane we may need a manoeuvre
        if self.planner.local_lane[0]['status'] == 'o' or self.planner.local_lane[0]['status'][-1] == 'l':
            return 0
        if isinstance(self.speed_plan, str) or self.speed_plan is None:
            return 0
        else:
            max_speed = np.max(self.speed_plan[:-1])
            min_speed = np.min(self.speed_plan[:-1])
            start_speed = self.speed_plan[0]
            print('speed plan max is :', max_speed)
            print('speed plan min is :', min_speed)
            print('speed plan start is :', start_speed)
            print('vehicle speed limit is :', self.planner.world.vehicle_speed_limit)
            print('score details :', start_speed - min_speed, '>', self.planner.world.vehicle_speed_limit / f0, 'or',
                  self.planner.world.vehicle_speed_limit - max_speed, '>', self.planner.world.vehicle_speed_limit / f1)
            if (start_speed - min_speed > self.planner.world.vehicle_speed_limit / f0
                    or self.planner.world.vehicle_speed_limit - max_speed
                    > self.planner.world.vehicle_speed_limit / f1):
                return 0
        return 1

    def renew_path(self, path, local_goal, ego_heading):
        if path is None:
            path = np.array([[self.ego_location[0], self.ego_location[1], ego_heading]])
        else:
            previous_path = path
            path = [[self.ego_location[0], self.ego_location[1], ego_heading]]
            fist_index, _ = get_index_offset(previous_path, self.ego_location, self.planner.world)
            previous_point = previous_path[fist_index]
            path.append([previous_point[0], previous_point[1], 0])
            for i, point in enumerate(previous_path):
                if i <= fist_index:
                    continue
                heading = np.arctan2(point[1] - previous_point[1], point[0] - previous_point[0])
                if heading < 0:
                    heading += 2 * math.pi
                if heading > 2 * math.pi:
                    heading -= 2 * math.pi
                path[-1][2] = heading
                path.append([point[0], point[1], 0])
                previous_point = point
            path[-1][2] = path[-2][2]
        self.static_path = get_static_path(path, local_goal, self.planner.local_lane, self.planner.world)
