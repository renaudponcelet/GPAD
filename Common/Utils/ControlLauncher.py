import carla
import numpy as np
from Common.Utils.utils import find_nearest_vector, SpeedPlan
from Common.Utils.agents.tools.misc import get_speed

class Controller:
    def __init__(self, planner, dt):
        self.path = None
        self.speed_plan = None
        self.planner = planner
        self.hard_brake = 0.5
        self.control_time = dt
        self._control = None
        self.target_location = None
        self.target_speed = None
        self.double_update_flag = False

    def update(self, path, speed_plan):
        if not self.double_update_flag:
            self.path = path
            self.speed_plan = speed_plan
            if self.speed_plan is None:
                self.speed_plan = SpeedPlan("brake")
            if self.path is not None:
                heading = np.array([
                    np.cos(np.deg2rad(self.planner.world.vehicle.get_transform().rotation.yaw)),
                    np.sin(np.deg2rad(self.planner.world.vehicle.get_transform().rotation.yaw))
                ])
                ego_pos = np.array(
                    [self.planner.world.vehicle.get_location().x, self.planner.world.vehicle.get_location().y]
                )
                axle_pos = ego_pos + self.planner.L_control * heading
                target_indexes = find_nearest_vector(self.path, axle_pos)
                target_inx = max(target_indexes)
                if target_inx == len(self.path) - 1:
                    target_point = axle_pos
                    self.speed_plan = SpeedPlan("brake")
                else:
                    target_point = self.path[target_inx]
                if np.dot(np.add(target_point, - ego_pos), heading) < 0:
                    target_point = axle_pos
                    self.speed_plan = SpeedPlan("brake")
                self.target_location = carla.Location(x=target_point[0], y=target_point[1])
                if self.planner.world.display:
                    self.planner.world.occupancy_viewer_ris.add_point(
                        target_point,
                        color='#ff00ff',
                        frame=self.planner.world.recorder.current_frame,
                        label="target location",
                        screens=0
                    )
                if isinstance(self.speed_plan, SpeedPlan):
                    if self.speed_plan == SpeedPlan("speed_up"):
                        self.target_speed = self.planner.world.vehicle_speed_limit * 3.6
                    elif self.speed_plan == SpeedPlan("keep_speed"):
                        self.target_speed = get_speed(self.planner.world.vehicle)
                    elif self.speed_plan == SpeedPlan("brake"):
                        self.target_speed = 0.0
                    else:
                        raise Exception("Speed mode is not enable !")
                else:
                    pre_target_speed = 3.6 * self.speed_plan[target_inx]
                    if pre_target_speed > self.planner.world.vehicle_speed_limit * 3.6:
                        self.target_speed = self.planner.world.vehicle_speed_limit * 3.6
                    elif pre_target_speed < 0:
                        self.target_speed = 0
                    else:
                        self.target_speed = pre_target_speed
                self._control = self.planner.PID_controller.run_step(
                    self.target_speed,
                    self.target_location,
                )
            else:
                if self._control is None:
                    self._control = carla.VehicleControl()
                self.get_brake_control(self._control)
            self.double_update_flag = True

    def run(self):
        self.planner.world.vehicle.apply_control(self._control)
        self.double_update_flag = False

    def get_brake_control(self, control):
        control.throttle = 0
        control.brake = self.hard_brake
