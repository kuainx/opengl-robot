import os
import struct
import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import transforms3d
from OpenGL.arrays import vbo
from OpenGL.GL import (
    GL_COLOR_ARRAY,
    GL_DIFFUSE,
    GL_FLOAT,
    GL_FRONT,
    GL_LIGHTING,
    GL_LINES,
    GL_NORMAL_ARRAY,
    GL_TRIANGLES,
    GL_VERTEX_ARRAY,
    glColorPointer,
    glDisable,
    glDisableClientState,
    glDrawArrays,
    glEnable,
    glEnableClientState,
    glLineWidth,
    glMaterialfv,
    glMultMatrixf,
    glNormalPointer,
    glPopMatrix,
    glPushMatrix,
    glRotatef,
    glVertexPointer,
)

Vertices = List[Tuple[float, float, float]]
Normals = List[Tuple[float, float, float]]


def parse_stl(filename: str) -> Tuple[Vertices, Normals]:
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
    def __init__(
        self, origin, vertices: Vertices, normals: Normals, color=(0.8, 0.8, 0.8, 1.0)
    ):
        self.origin = origin
        self.color = color
        # 转换为numpy数组并确保形状正确
        vertices_np = np.array(vertices, dtype="f4").reshape(-1, 3)
        normals_np = np.array(normals, dtype="f4").reshape(-1, 3)
        self.vertex_vbo = vbo.VBO(vertices_np)
        self.normal_vbo = vbo.VBO(normals_np)
        self.vertex_count = vertices_np.shape[0]  # 直接使用顶点数量

    def draw(self):
        glPushMatrix()
        glMaterialfv(GL_FRONT, GL_DIFFUSE, self.color)  # 设置材质颜色
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
        self.mimic_joint: str | None = None  # 被模仿的关节名称
        self.multiplier = 1.0  # 比例系数
        self.offset = 0.0  # 偏移量


def parse_urdf(urdf_file: str) -> Tuple[Link, List[Joint]]:
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
                stl_filename = mesh_elem.get("filename") or ""
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

        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            print(f"Warning: Joint {joint_name} is missing parent or child element")
            continue
        parent_link = links[parent_elem.get("link")]
        child_link = links[child_elem.get("link")]
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
        mimic_elem = joint_elem.find("mimic")
        if mimic_elem is not None:
            joint.mimic_joint = mimic_elem.get("joint")
            joint.multiplier = float(mimic_elem.get("multiplier", "1.0"))
            joint.offset = float(mimic_elem.get("offset", "0.0"))
        parent_link.children.append(joint)
        if joint_type == "revolute":
            revolute_joints.append(joint)  # 收集可旋转关节

    # 确定根link（未被任何joint作为child引用的link）
    child_links = set(joint.child.name for joint in revolute_joints)
    root_links = [link for link in links.values() if link.name not in child_links]
    return root_links[0], revolute_joints


def create_cuboid(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float):
    vertices: Vertices = []
    normals: Normals = []

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


def create_sphere(radius=0.05, slices=16, stacks=16):
    vertices = []
    normals = []
    for i in range(slices):
        theta1 = i * 2 * np.pi / slices
        theta2 = (i + 1) * 2 * np.pi / slices
        for j in range(stacks):
            phi1 = j * np.pi / stacks
            phi2 = (j + 1) * np.pi / stacks
            # 生成四个顶点
            p1 = spherical_to_cartesian(theta1, phi1, radius)
            p2 = spherical_to_cartesian(theta2, phi1, radius)
            p3 = spherical_to_cartesian(theta2, phi2, radius)
            p4 = spherical_to_cartesian(theta1, phi2, radius)
            # 添加两个三角形
            vertices.extend([p1, p2, p3, p1, p3, p4])
            # 生成法线
            n1 = [x / radius for x in p1]
            n2 = [x / radius for x in p2]
            n3 = [x / radius for x in p3]
            n4 = [x / radius for x in p4]
            normals.extend([n1, n2, n3, n1, n3, n4])
    return vertices, normals


def spherical_to_cartesian(theta, phi, r):
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return (x, y, z)


class CoordinateAxis:
    def __init__(self, axis_length=0.15):
        # 创建坐标系VBO
        self.axis_length = axis_length
        self.vertices, self.colors = self.create_axis_data()

        # 创建VBO
        self.vertex_vbo = vbo.VBO(np.array(self.vertices, dtype="f4"))
        self.color_vbo = vbo.VBO(np.array(self.colors, dtype="f4"))

    def create_axis_data(self):
        vertices = []
        colors = []
        # X轴（红色）
        vertices.extend([(0, 0, 0), (self.axis_length, 0, 0)])
        colors.extend([(1, 0, 0, 1), (1, 0, 0, 1)])
        # Y轴（绿色）
        vertices.extend([(0, 0, 0), (0, self.axis_length, 0)])
        colors.extend([(0, 1, 0, 1), (0, 1, 0, 1)])
        # Z轴（蓝色）
        vertices.extend([(0, 0, 0), (0, 0, self.axis_length)])
        colors.extend([(0, 0, 1, 1), (0, 0, 1, 1)])
        return vertices, colors

    def draw(self, transform):
        glPushMatrix()
        glMultMatrixf(transform.T.ravel())

        # 禁用光照计算
        glDisable(GL_LIGHTING)

        # 设置顶点颜色
        self.color_vbo.bind()
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_FLOAT, 0, None)

        # 设置顶点位置
        self.vertex_vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)

        # 绘制线段
        glLineWidth(3)
        glDrawArrays(GL_LINES, 0, len(self.vertices))
        glLineWidth(1)

        # 恢复状态
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        self.vertex_vbo.unbind()
        self.color_vbo.unbind()
        glEnable(GL_LIGHTING)
        glPopMatrix()


class Target:
    def __init__(self):
        self.axis = CoordinateAxis()
        self.transform = np.eye(4)

    def update_pose(self, position, orientation):
        self.transform[:3, :3] = orientation
        self.transform[:3, 3] = position

    def draw(self):
        self.axis.draw(self.transform)


class Floor:
    def __init__(
        self,
        size=5.0,
        step=1.0,
        fill_color=(0.9, 0.9, 0.9, 1.0),
        line_color=(0.7, 0.7, 0.7, 1.0),
    ):
        # 生成填充面和网格线数据
        self.fill_vertices, self.fill_colors = self.create_fill_data(size, fill_color)
        self.line_vertices, self.line_colors = self.create_grid_data(
            size, step, line_color
        )

        # 创建VBO
        self.fill_vertex_vbo = vbo.VBO(np.array(self.fill_vertices, dtype="f4"))
        self.fill_color_vbo = vbo.VBO(np.array(self.fill_colors, dtype="f4"))
        self.line_vertex_vbo = vbo.VBO(np.array(self.line_vertices, dtype="f4"))
        self.line_color_vbo = vbo.VBO(np.array(self.line_colors, dtype="f4"))

    def create_fill_data(self, size, color):
        """创建地板填充面数据（两个三角形组成矩形）"""
        vertices = [
            (-size, -size, 0),
            (size, -size, 0),
            (size, size, 0),
            (-size, -size, 0),
            (size, size, 0),
            (-size, size, 0),
        ]
        colors = [color] * 6  # 每个顶点使用相同颜色
        return vertices, colors

    def create_grid_data(self, size, step, color):
        """创建网格线数据"""
        vertices = []
        colors = []
        # 横向线条（沿X轴方向）
        for y in np.arange(-size, size + step, step):
            vertices.extend([(-size, y, 0.001), (size, y, 0.001)])
            colors.extend([color, color])
        # 纵向线条（沿Y轴方向）
        for x in np.arange(-size, size + step, step):
            vertices.extend([(x, -size, 0.001), (x, size, 0.001)])
            colors.extend([color, color])
        return vertices, colors

    def draw(self):
        glDisable(GL_LIGHTING)  # 禁用光照
        glPushMatrix()

        # 先绘制填充面
        self.fill_color_vbo.bind()
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(4, GL_FLOAT, 0, None)

        self.fill_vertex_vbo.bind()
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, None)
        glDrawArrays(GL_TRIANGLES, 0, len(self.fill_vertices))

        # 再绘制网格线
        self.line_color_vbo.bind()
        glColorPointer(4, GL_FLOAT, 0, None)

        self.line_vertex_vbo.bind()
        glVertexPointer(3, GL_FLOAT, 0, None)
        glLineWidth(1)
        glDrawArrays(GL_LINES, 0, len(self.line_vertices))

        # 清理状态
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)
        self.fill_vertex_vbo.unbind()
        self.fill_color_vbo.unbind()
        self.line_vertex_vbo.unbind()
        self.line_color_vbo.unbind()

        glEnable(GL_LIGHTING)
        glPopMatrix()
