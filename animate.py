import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R


class AnimationController:
    class State:
        Idle = "idle"
        MovingToPick = "moving_to_pick"
        Grabbing = "grabbing"
        MoveJ = "movej"
        MovingToPlace = "moving_to_place"
        Releasing = "releasing"

    def __init__(self, robot, shelf, receive_position):
        self.robot = robot
        self.shelf = shelf
        self.receive_position = receive_position

        self.state = self.State.Idle
        self.grabbed_object = None
        self.path = []
        self.current_step = 0
        self.start_time = 0.0
        self.last_frame_time = 0
        self.start_joint = None
        self.end_joint = None

    def draw_obj(self):
        if self.grabbed_object:
            if self.current_step > 2:
                self._update_grabbed_position()
            self.grabbed_object.draw()

    def start_pick_and_place(self):
        if self.state != self.State.Idle or not self.shelf.objects:
            return False

        self.grabbed_object = self.shelf.objects.pop(0)
        grab_pos = self.grabbed_object.origin[:3, 3].copy()
        grab_pre = grab_pos + np.array([-0.1, 0, 0])
        receive_pre = self.receive_position + np.array([0, 0, 0.1])
        grab_rot = R.from_euler("yx", [np.pi / 2, np.pi / 2]).as_matrix()
        release_rot = R.from_euler("yx", [-np.pi / 2, np.pi / 2]).as_matrix()
        self.robot.move_joint(6, np.radians(50))
        # 路径点包含位置和旋转
        self.path = [
            (self.robot.current_position, self.robot.current_orientation, 1),
            (grab_pre, grab_rot, 10),
            (grab_pos, grab_rot, 2),
            (grab_pre, grab_rot, 2),
            (receive_pre, release_rot, 10),
            (self.receive_position, release_rot, 2),
            (receive_pre, release_rot, 2),
        ]

        self.state = self.State.MovingToPick
        self.current_step = 0
        self.start_time = glfw.get_time()
        return True

    def update(self):
        if self.state == self.State.Idle:
            return
        if self.state == self.State.MovingToPick:
            self.robot.move_joint(6, np.radians(50))
            self._handle_movement(
                start_step=0,
                end_step=3,
                next_state=self.State.Grabbing,
            )
        elif self.state == self.State.Grabbing:
            self.robot.move_joint(6, np.radians(30))
            self._handle_movement(
                start_step=4,
                end_step=4,
                next_state=self.State.MoveJ,
            )
        elif self.state == self.State.MoveJ:
            self._handle_movement(
                start_step=5,
                end_step=5,
                next_state=self.State.MovingToPlace,
            )
        elif self.state == self.State.MovingToPlace:
            self._handle_movement(
                start_step=6,
                end_step=6,
                next_state=self.State.Releasing,
            )
        elif self.state == self.State.Releasing:
            self.robot.move_joint(6, np.radians(50))
            self.grabbed_object.origin[:3, 3] = self.receive_position
            self._reset_state()

    # 修改动画控制器中的旋转插值方法
    def _interpolate_rotation(self, start_rot, end_rot, t):
        """使用四元数进行球面线性插值"""
        r_start = R.from_matrix(start_rot).as_quat()
        r_end = R.from_matrix(end_rot).as_quat()
        # 手动实现slerp
        dot = np.dot(r_start, r_end)
        theta = np.arccos(np.clip(dot, -1, 1))
        sin_theta = np.sin(theta)

        if sin_theta < 1e-6:  # 避免除以零
            return R.from_quat(r_start).as_matrix()

        a = np.sin((1 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta
        interp_quat = a * r_start + b * r_end
        return R.from_quat(interp_quat).as_matrix()

    def _handle_movement(self, start_step, end_step, next_state):
        if self.current_step < end_step:
            idx = self.current_step
            start_pos, start_rot, _ = (
                self.path[idx] if idx == start_step else self.path[idx - 1]
            )
            end_pos, end_rot, duration = self.path[idx]

            t = min((glfw.get_time() - self.start_time) / duration, 1.0)
            print(t)
            new_pos = start_pos * (1 - t) + end_pos * t  # 线性插值
            new_rot = self._interpolate_rotation(start_rot, end_rot, t)

            self.robot.update_joint_angles(new_pos, new_rot)

            if t >= 1.0:
                self.current_step += 1
                self.start_time = glfw.get_time()
        else:
            self.state = next_state
            self.start_time = glfw.get_time()

    def _update_grabbed_position(self):
        """同步被抓物体到机械臂末端"""
        if self.grabbed_object:
            self.grabbed_object.origin[:3, 3] = self.robot.current_position
            self.grabbed_object.origin[:3, :3] = self.robot.current_orientation

    def _reset_state(self):
        """重置动画控制器状态"""
        self.state = self.State.Idle
        self.grabbed_object = None
        self.path = []
        self.current_step = 0
        self.start_time = 0.0
        self.last_frame_time = 0
        self.start_joint = None
        self.end_joint = None

    @property
    def is_animating(self):
        return self.state != self.State.Idle
