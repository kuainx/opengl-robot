import glfw
import numpy as np
from scipy.spatial.transform import Rotation as R


class AnimationController:
    class State:
        Idle = "idle"
        MovingToPick = "moving_to_pick"
        Grabbing = "grabbing"
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
        self.duration = 1.0  # 默认阶段持续时间

    def start_pick_and_place(self):
        if self.state != self.State.Idle or not self.shelf.objects:
            return False

        self.grabbed_object = self.shelf.objects.pop(0)
        grab_pos = self.grabbed_object.origin[:3, 3].copy()
        self.grab_rot = R.from_euler("yx", [np.pi / 2, np.pi / 2]).as_matrix()
        self.release_rot = R.from_euler("yx", [-np.pi / 2, np.pi / 2]).as_matrix()

        # 路径点包含位置和旋转
        self.path = [
            (grab_pos + np.array([-0.1, 0, 0]), self.grab_rot),  # 预抓取点
            (grab_pos, self.grab_rot),  # 抓取点（平放）
            (self.receive_position + np.array([0, 0, 0.1]), self.release_rot),
            (self.receive_position, self.release_rot),
        ]

        self.state = self.State.MovingToPick
        self.current_step = 0
        self.start_time = glfw.get_time()
        self.duration = 2.0
        return True

    def update(self):
        if self.state == self.State.Idle:
            return

        current_time = glfw.get_time()
        elapsed = current_time - self.start_time
        t = min(elapsed / self.duration, 1.0)

        if self.state == self.State.MovingToPick:
            self._handle_movement(
                start_step=0,
                end_step=2,  # 移动到抓取点并返回预放置
                next_state=self.State.Grabbing,
            )

            # 更新被抓物体位置（仅在实际抓取阶段）
            if self.current_step == 1:
                self._update_grabbed_position()

        elif self.state == self.State.Grabbing:
            # 设置第七关节为闭合角度
            self.robot.revolute_joints[6].angle = np.radians(30)
            if t >= 1.0:
                self.state = self.State.MovingToPlace
                self.current_step = 2  # 从预放置开始
                self.start_time = current_time
                self.duration = 2.0

        elif self.state == self.State.MovingToPlace:
            self._handle_movement(
                start_step=2,
                end_step=4,  # 移动到放置位置
                next_state=self.State.Releasing,
            )

        elif self.state == self.State.Releasing:
            if t >= 1.0:
                # 完成放置后重置状态
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
            start_pos, start_rot = (
                self.path[idx] if idx == start_step else self.path[idx - 1]
            )
            end_pos, end_rot = self.path[idx]

            t = min((glfw.get_time() - self.start_time) / self.duration, 1.0)
            new_pos = start_pos + (end_pos - start_pos) * t
            new_rot = self._interpolate_rotation(start_rot, end_rot, t)

            self.robot.update_joint_angles(new_pos, new_rot)

            if t >= 1.0:
                self.current_step += 1
                self.start_time = glfw.get_time()
        else:
            self.state = next_state
            self.start_time = glfw.get_time()

    def _cubic_bezier(self, p0, p1, p2, p3, t):
        """三次贝塞尔曲线插值"""
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2
            + t**3 * p3
        )

    def _update_grabbed_position(self):
        """同步被抓物体到机械臂末端"""
        if self.grabbed_object:
            self.grabbed_object.origin[:3, 3] = self.robot.current_position

    def _reset_state(self):
        """重置动画控制器状态"""
        self.state = self.State.Idle
        self.grabbed_object = None
        self.path = []
        self.current_step = 0

    @property
    def is_animating(self):
        return self.state != self.State.Idle
