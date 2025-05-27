import numpy as np
from ikpy import chain


class Robot:
    def __init__(self, urdf_path, ik_urdf, revolute_joints):
        self.urdf_path = urdf_path
        self.ik_chain = chain.Chain.from_urdf_file(
            ik_urdf,
            base_elements=["base_link"],
            # active_links_mask=[False] + [True] * 5 + [False] * 4,
        )
        self.revolute_joints = revolute_joints
        self.current_position = np.zeros(3)
        self.current_orientation = np.eye(3)

    def move_joint(self, i, angle, relative=False):
        if relative:
            self.revolute_joints[i].angle += angle
        else:
            self.revolute_joints[i].angle = angle
        for joint in self.revolute_joints:
            if joint.mimic_joint is not None:
                master_joint = next(
                    (j for j in self.revolute_joints if j.name == joint.mimic_joint),
                    None,
                )
                if master_joint:
                    joint.angle = master_joint.angle * joint.multiplier + joint.offset
        self.current_position, self.current_orientation = self.update_pos()

    def update_joint_angles(self, target_pos, target_rot):
        initial_angles = [0.0] + [joint.angle for joint in self.revolute_joints]
        try:
            joint_angles = self.ik_chain.inverse_kinematics(
                target_pos, target_rot, "all", initial_position=initial_angles[:7]
            )
            for i in range(1, 7):
                self.revolute_joints[i - 1].angle = joint_angles[i]
            self.current_position, self.current_orientation = self.update_pos()
        except Exception as e:
            print(f"IK error: {str(e)}")

    def update_pos(self):
        initial_angles = [0.0] + [joint.angle for joint in self.revolute_joints]
        T = np.eye(4)
        T[:4, :4] = self.ik_chain.forward_kinematics(
            initial_angles[:7] + [np.radians(40)] * 0
        )
        pos = T[:3, 3]
        rot = T[:3, :3]
        return pos, rot
