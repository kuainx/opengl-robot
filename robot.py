import numpy as np
from ikpy import chain


class Robot:
    def __init__(self, urdf_path, revolute_joints):
        self.urdf_path = urdf_path
        self.ik_chain = chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=["base_link"],
            active_links_mask=[False] + [True] * 6 + [False] * 3,
        )
        self.revolute_joints = revolute_joints

    def update_joint_angles(self, target_pos, target_rot):
        # 获取当前关节角度（排除固定关节）
        # initial_angles = [0.0] + [joint.angle for joint in self.revolute_joints]
        try:
            # 求解逆运动学
            joint_angles = self.ik_chain.inverse_kinematics(
                target_pos, target_rot, "all"
            )
            # 更新关节角度（排除第一个固定关节）
            for i in range(1, 7):
                self.revolute_joints[i - 1].angle = joint_angles[i]  # 跳过基座关节

        except Exception as e:
            print(f"IK求解失败: {str(e)}")

    def update_pos(self):
        initial_angles = [0.0] + [joint.angle for joint in self.revolute_joints]
        T = np.eye(4)
        T[:4, :4] = self.ik_chain.forward_kinematics(initial_angles[:7] + [0] * 3)
        pos = T[:3, 3]
        rot = T[:3, :3]
        return pos, rot
