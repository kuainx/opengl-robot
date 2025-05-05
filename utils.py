import transforms3d
import numpy as np
import xml.etree.ElementTree as ET
import struct
import os
from OpenGL.arrays import vbo
from OpenGL.GL import *


def parse_stl(filename):
    """解析STL文件, 支持ASCII和二进制格式, 返回顶点和法线列表"""
    try:
        with open(filename, "rb") as f:
            header = f.read(80)
            num_triangles = struct.unpack("<I", f.read(4))[0]
            expected_size = 84 + num_triangles * 50
            if os.path.getsize(filename) == expected_size:
                return parse_binary(f, num_triangles)
    except:
        pass
    return parse_ascii(filename)


def parse_binary(f, num_triangles):
    """解析二进制STL文件"""
    vertices = []
    normals = []
    for _ in range(num_triangles):
        data = f.read(50)
        normal = struct.unpack("<3f", data[0:12])
        v1, v2, v3 = (
            struct.unpack("<3f", data[12:24]),
            struct.unpack("<3f", data[24:36]),
            struct.unpack("<3f", data[36:48]),
        )
        vertices.extend([v1, v2, v3])
        normals.extend([normal] * 3)
    return vertices, normals


def parse_ascii(filename):
    """解析ASCII STL文件"""
    vertices = []
    normals = []
    with open(filename, "r") as f:
        current_normal = None
        for line in f:
            line = line.strip()
            if line.startswith("facet normal"):
                current_normal = list(map(float, line.split()[2:5]))
            elif line.startswith("vertex"):
                vertex = list(map(float, line.split()[1:4]))
                vertices.append(vertex)
                normals.append(current_normal)
    return vertices, normals


class Visual:
    def __init__(self, origin, vertices, normals):
        self.origin = origin
        # 转换为numpy数组并确保形状正确
        vertices_np = np.array(vertices, dtype="f4").reshape(-1, 3)
        normals_np = np.array(normals, dtype="f4").reshape(-1, 3)
        self.vertex_vbo = vbo.VBO(vertices_np)
        self.normal_vbo = vbo.VBO(normals_np)
        self.vertex_count = vertices_np.shape[0]  # 直接使用顶点数量

    def draw(self):
        glPushMatrix()
        glMultMatrixf(self.origin.T.ravel())  # 列主序转换

        self.vertex_vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)

        self.normal_vbo.bind()
        glEnableClientState(GL_NORMAL_ARRAY)
        glNormalPointer(GL_FLOAT, 0, None)

        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)

        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        self.normal_vbo.unbind()
        self.vertex_vbo.unbind()

        glPopMatrix()


class Link:
    def __init__(self, name):
        self.name = name
        self.visuals = []
        self.children = []  # 子关节列表

    def draw(self):
        for visual in self.visuals:
            visual.draw()
        for joint in self.children:
            glPushMatrix()
            # 应用关节原始变换
            glMultMatrixf(joint.origin.T.ravel())
            # 应用关节旋转（仅限旋转关节）
            if joint.type == "revolute":
                angle_deg = np.degrees(joint.angle)
                glRotatef(angle_deg, *joint.axis)
            # 递归绘制子link
            joint.child.draw()
            glPopMatrix()


class Joint:
    def __init__(self, name, joint_type, origin, child, axis):
        self.name = name
        self.type = joint_type  # 关节类型（revolute/prismatic等）
        self.origin = origin  # 原始变换矩阵
        self.child = child  # 子link
        self.axis = axis  # 旋转/移动轴
        self.angle = 0.0  # 当前关节角度（弧度）


def parse_urdf(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()

    links = {}
    # 解析所有link
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name")
        link = Link(link_name)
        # 解析visual部分
        for visual_elem in link_elem.findall("visual"):
            # 解析origin
            origin_elem = visual_elem.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin_elem is not None:
                xyz = list(map(float, origin_elem.get("xyz", "0 0 0").split()))
                rpy = list(map(float, origin_elem.get("rpy", "0 0 0").split()))
            # 计算变换矩阵
            rotation = transforms3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], "sxyz")
            origin_matrix = np.eye(4)
            origin_matrix[:3, :3] = rotation
            origin_matrix[:3, 3] = xyz
            # 解析geometry
            mesh_elem = visual_elem.find("geometry/mesh")
            if mesh_elem is not None:
                stl_filename = mesh_elem.get("filename")
                stl_filename = os.path.join(os.path.dirname(urdf_file), stl_filename)
                vertices, normals = parse_stl(stl_filename)
                link.visuals.append(Visual(origin_matrix, vertices, normals))
        links[link_name] = link

    # 解析所有joint
    revolute_joints = []
    for joint_elem in root.findall("joint"):
        joint_name = joint_elem.get("name")
        joint_type = joint_elem.get("type")
        axis_elem = joint_elem.find("axis")
        axis = [0.0, 0.0, 1.0]  # 默认z轴
        if axis_elem is not None:
            axis = list(map(float, axis_elem.get("xyz", "0 0 1").split()))

        parent_link = links[joint_elem.find("parent").get("link")]
        child_link = links[joint_elem.find("child").get("link")]
        # 解析origin
        origin_elem = joint_elem.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin_elem is not None:
            xyz = list(map(float, origin_elem.get("xyz", "0 0 0").split()))
            rpy = list(map(float, origin_elem.get("rpy", "0 0 0").split()))
        rotation = transforms3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], "sxyz")
        origin_matrix = np.eye(4)
        origin_matrix[:3, :3] = rotation
        origin_matrix[:3, 3] = xyz
        # 创建joint并添加到父link的子列表
        joint = Joint(joint_name, joint_type, origin_matrix, child_link, axis)
        parent_link.children.append(joint)
        if joint_type == "revolute":
            revolute_joints.append(joint)  # 收集可旋转关节

    # 确定根link（未被任何joint作为child引用的link）
    child_links = set(joint.child.name for joint in revolute_joints)
    root_links = [link for link in links.values() if link.name not in child_links]
    return root_links[0] if root_links else None, revolute_joints


def create_cuboid(x1, y1, z1, x2, y2, z2):
    vertices = []
    normals = []

    # 前面 (z = z2)
    vertices.extend([(x1, y1, z2), (x2, y1, z2), (x2, y2, z2)])
    vertices.extend([(x1, y1, z2), (x2, y2, z2), (x1, y2, z2)])
    normals += [(0, 0, 1)] * 6

    # 后面 (z = z1)
    vertices.extend([(x1, y1, z1), (x2, y2, z1), (x2, y1, z1)])
    vertices.extend([(x1, y1, z1), (x1, y2, z1), (x2, y2, z1)])
    normals += [(0, 0, -1)] * 6

    # 左面 (x = x1)
    vertices.extend([(x1, y1, z1), (x1, y2, z1), (x1, y2, z2)])
    vertices.extend([(x1, y1, z1), (x1, y2, z2), (x1, y1, z2)])
    normals += [(-1, 0, 0)] * 6

    # 右面 (x = x2)
    vertices.extend([(x2, y1, z1), (x2, y1, z2), (x2, y2, z2)])
    vertices.extend([(x2, y1, z1), (x2, y2, z2), (x2, y2, z1)])
    normals += [(1, 0, 0)] * 6

    # 顶面 (y = y2)
    vertices.extend([(x1, y2, z1), (x2, y2, z1), (x2, y2, z2)])
    vertices.extend([(x1, y2, z1), (x2, y2, z2), (x1, y2, z2)])
    normals += [(0, 1, 0)] * 6

    # 底面 (y = y1)
    vertices.extend([(x1, y1, z1), (x2, y1, z2), (x2, y1, z1)])
    vertices.extend([(x1, y1, z1), (x1, y1, z2), (x2, y1, z2)])
    normals += [(0, -1, 0)] * 6

    return vertices, normals
