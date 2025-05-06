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
            for i, joint in enumerate(self.revolute_joints):
                joint.angle = joint_angles[i + 1]  # 跳过基座关节

        except Exception as e:
            print(f"IK求解失败: {str(e)}")
