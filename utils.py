import transforms3d
import numpy as np
import xml.etree.ElementTree as ET
import struct
import os


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
        self.origin = origin  # 4x4变换矩阵
        self.vertices = vertices
        self.normals = normals


class Link:
    def __init__(self, name):
        self.name = name
        self.visuals = []
        self.children = []  # 子关节列表


class Joint:
    def __init__(self, name, origin, child):
        self.name = name
        self.origin = origin  # 4x4变换矩阵
        self.child = child  # 子Link


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
    joints = []
    for joint_elem in root.findall("joint"):
        joint_name = joint_elem.get("name")
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
        joint = Joint(joint_name, origin_matrix, child_link)
        parent_link.children.append(joint)
        joints.append(joint)

    # 确定根link（未被任何joint作为child引用的link）
    child_links = set(joint.child.name for joint in joints)
    root_links = [link for link in links.values() if link.name not in child_links]
    return root_links[0] if root_links else None
