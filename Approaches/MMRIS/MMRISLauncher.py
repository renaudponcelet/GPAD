import math
import threading

import numpy as np
from ...Common.Utils.carla_utils import get_static_path, carla_vector2array_2d, get_index_offset
from ...Common.Utils.utils import SpeedPlan
from ...Common.Utils.agents.tools.misc import get_speed


class GetPath(threading.Thread):
    def __init__(self, planner, static_path, speed_plan, ris, goal):
        super(GetPath, self).__init__()
        self.static_path = static_path
        self.path = None
        self.ris = ris
        self.speed_plan = speed_plan
        self.planner = planner
        self.goal = goal

    def run(self):
        self.path = self.ris.get_path(
            planner=self.planner,
            static_path=self.static_path,
            speed_profile=self.speed_plan,
            goal=self.goal
        )


class RISLauncher:
    name = "launcher for MMRIS thread"

    def __init__(self, planner):
        self.planner = planner
        self.quality_flag = True
        self.path = None
        self.speed_plan = None
        self.ris_threads = {}
        self.time_since_replan = 0
        self.time_at_last_replan = 0
        self.launched_plan = []
        self.low_quality_plan = []
        self.saved_ris = []
        self.ego_location = None
        self.static_path = None
        self.speed_plans = [SpeedPlan("speed_up"), SpeedPlan("keep_speed"), SpeedPlan("brake")]

    def launch(self, local_goal, path=None):
        if self.planner.counter % (self.planner.world.replan_ratio + 1) == 0:
            speed_ratio = (get_speed(self.planner.world.vehicle) / 3.6) / self.planner.world.vehicle_speed_limit
            self.time_since_replan = 0
            self.saved_ris = []
            self.ego_location = carla_vector2array_2d(
                self.planner.world.vehicle.get_transform().location).round(decimals=2)
            ego_heading = np.deg2rad(self.planner.world.vehicle.get_transform().rotation.yaw)
            self.time_at_last_replan = self.planner.world.world_time
            self.planner.world.occupancy_mapper.update_obstacles()
            self.renew_path(path, local_goal, ego_heading)
            stopped_at_a_red_light = False
            if (str(self.planner.world.occupancy_mapper.traffic_light_state) == 'Red'
                    and get_speed(self.planner.world.vehicle) < 2):
                print('ego is stopped at a red light')
                stopped_at_a_red_light = True
            if self.planner.world.occupancy_mapper.in_map(self.static_path,
                                                          self.planner.world.vehicle.bounding_box.extent.y):
                print("static_path in map restart from ego location")
                self.planner.world.occupancy_viewer_ris.add_line(self.static_path, color='#0f0f00',
                                                                 frame=self.planner.world.recorder.current_frame,
                                                                 screens=[0, 1, 2, 3],
                                                                 label='static path')
                path = np.array([[self.ego_location[0], self.ego_location[1], ego_heading]])
                self.static_path = get_static_path(path, local_goal, self.planner.local_lane, self.planner.world)
            self.planner.world.occupancy_mapper.add_time_gap_to_obstacles(self.planner.world.time_gap)
            self.planner.world.occupancy_mapper.add_dynamic_signal_to_obstacles()
            self.launched_plan = []
            for speed_plan in self.speed_plans:
                if stopped_at_a_red_light:
                    continue
                if speed_ratio > 0.8 and speed_plan == 'speed_up':
                    continue
                if speed_ratio < 0.1 and (speed_plan == 'keep_speed' or speed_plan == 'brake'):
                    continue
                if self.planner.world.allow_threading:
                    self.ris_threads[speed_plan] = GetPath(
                        planner=self.planner,
                        static_path=self.static_path,
                        speed_plan=speed_plan,
                        ris=self.planner.world.MMRIS,
                        goal=local_goal
                    )
                    self.ris_threads[speed_plan].start()
                self.launched_plan.append(speed_plan)

    def call(self, goal=None, display_only=False):
        if self.planner.counter % (self.planner.world.replan_ratio + 1) == 0 or display_only:
            if not display_only:
                self.quality_flag = True
                self.low_quality_plan = []
                for speed_plan in self.launched_plan:
                    print("speed plan :", speed_plan)
                    if self.planner.world.display:
                        self.planner.world.occupancy_viewer_ris.add_circle(
                            [- 2 * self.planner.world.visual_horizon + self.ego_location[0] + 2,
                             - 2 * self.planner.world.visual_horizon + self.ego_location[1] + 2,
                             1],
                            color='#ff0000',
                            frame=self.planner.world.recorder.current_frame,
                            screens=int(speed_plan),
                        )
                    if self.planner.world.allow_threading:
                        self.ris_threads[speed_plan].join(timeout=6)
                        temp_path = self.ris_threads[speed_plan].path
                    else:
                        temp_path = self.planner.world.MMRIS.get_path(
                            planner=self.planner,
                            static_path=self.static_path,
                            speed_profile=speed_plan,
                            goal=goal
                        )
                    if temp_path is not None:
                        self.path = temp_path
                        self.speed_plan = speed_plan
                        self.quality_flag = True
                        break
                    else:
                        self.quality_flag = False
                        self.path = None
            if self.planner.world.display:
                if self.path is not None and not isinstance(self.path, str):
                    self.planner.world.occupancy_viewer_ris.add_line(self.path, color='#ff0000',
                                                                     frame=self.planner.world.recorder.current_frame,
                                                                     screens=0,
                                                                     label='final path')
                if goal is not None:
                    self.planner.world.occupancy_viewer_ris.add_circle([goal[0], goal[1], 0.5],
                                                                       color='#000000',
                                                                       frame=self.planner.world.recorder.current_frame,
                                                                       screens=[0, 1, 2, 3], label='goal')
                if self.planner.rec["first_time"] is None:
                    self.planner.rec["first_time"] = [0]
            self.planner.controller.update(self.path, self.speed_plan)
            self.draw()
            self.time_since_replan = 0
            self.time_at_last_replan = self.planner.world.world_time
        elif self.planner.world.planning_mode is "parallel" and self.planner.selected_plan != 'ris-path':
            self.path = None

    def draw(self):
        pos = self.planner.world.vehicle.get_location()
        self.planner.world.occupancy_viewer_ris.show_figure(
            frame=self.planner.world.recorder.current_frame,
            ref_time=self.planner.rec["first_time"][0],
            flip=True,
            rec_folder=self.planner.world.rec_folder,
            world=self.planner.world,
            title='MMRIS results',
            mode=self.speed_plan,
            x_lim=[- 2 * self.planner.world.visual_horizon + pos.x, 2 * self.planner.world.visual_horizon + pos.x],
            x_label='x (meters)',
            y_lim=[- 2 * self.planner.world.visual_horizon + pos.y, 2 * self.planner.world.visual_horizon + pos.y],
            y_label='y (meters)',
            zoom=3,
            time=self.time_at_last_replan,
            screen=[0, 1, 2, 3]
        )

    def respawn(self):
        self.ris_threads = {}
        self.path = None
        self.speed_plan = None

    def get_results(self):
        return self.path, self.speed_plan

    def get_score(self):
        if self.path is None:
            return 0
        if self.quality_flag:
            return 1
        return 0

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
        self.static_path = get_static_path(np.array(path), local_goal, self.planner.local_lane, self.planner.world)
