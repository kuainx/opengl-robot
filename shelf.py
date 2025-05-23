import colorsys
import random

import numpy as np
from OpenGL.GL import GL_DIFFUSE, GL_FRONT, glMaterialfv

from utils import Visual, create_cuboid, create_sphere


class Shelf:
    def __init__(self, box_size, position):
        self.box_size = box_size
        self.position = position
        shelf_vertices, shelf_normals = self.create_visual()
        shelf_origin = np.eye(4)
        shelf_origin[:3, 3] = position  # 调整货架位置
        self.visual = Visual(
            shelf_origin, shelf_vertices, shelf_normals, color=(0.6, 0.6, 0.6, 1.0)
        )
        self.objects = []
        self.generate_objects()

    def create_visual(self):
        width, depth, height, num_shelves = self.box_size
        vertices = []
        normals = []
        pillar_size = 0.01
        corners = [
            (-width / 2, -depth / 2, 0),
            (width / 2, -depth / 2, 0),
            (width / 2, depth / 2, 0),
            (-width / 2, depth / 2, 0),
        ]

        for x, y, z in corners:
            x1, y1 = x - pillar_size / 2, y - pillar_size / 2
            x2, y2 = x + pillar_size / 2, y + pillar_size / 2
            pillar_verts, pillar_norms = create_cuboid(
                x1, y1, z - self.position[2], x2, y2, z + height
            )
            vertices.extend(pillar_verts)
            normals.extend(pillar_norms)

        shelf_thickness = 0.005
        shelf_height_step = height / (num_shelves + 1)
        for i in range(num_shelves):
            shelf_z = (i + 1) * shelf_height_step
            shelf_verts, shelf_norms = create_cuboid(
                -width / 2,
                -depth / 2,
                shelf_z - shelf_thickness / 2,
                width / 2,
                depth / 2,
                shelf_z + shelf_thickness / 2,
            )
            vertices.extend(shelf_verts)
            normals.extend(shelf_norms)

        return vertices, normals

    def generate_objects(self):
        width, depth, height, num_shelves = self.box_size
        shelf_thickness = 0.005
        shelf_height_step = height / (num_shelves + 1)
        for i in range(num_shelves):
            shelf_z_local = (i + 1) * shelf_height_step
            shelf_top_z = self.position[2] + shelf_z_local + shelf_thickness / 2
            num_objects = 3
            placed = []  # 存储该层已放置物体的信息 (x, y, radius)

            for _ in range(num_objects):
                max_attempts = 100  # 防止无限循环
                obj_placed = False

                for _ in range(max_attempts):
                    shape_type = random.choice(["cube", "sphere"])
                    obj_size = 0.05
                    half_size = obj_size / 2
                    effective_radius = half_size + 0.01  # 有效检测半径
                    # 调整可用空间（考虑间隔）
                    x_min = -width / 2 + effective_radius
                    x_max = width / 2 - effective_radius
                    y_min = -depth / 2 + effective_radius
                    y_max = depth / 2 - effective_radius

                    if x_max < x_min or y_max < y_min:  # 空间不足时跳过
                        continue

                    # 生成候选位置（确保在货架边界内）
                    x_local = random.uniform(
                        -width / 2 + half_size, width / 2 - half_size
                    )
                    y_local = random.uniform(
                        -depth / 2 + half_size, depth / 2 - half_size
                    )
                    x_global = self.position[0] + x_local
                    y_global = self.position[1] + y_local
                    z_global = shelf_top_z + half_size

                    # 碰撞检测（视为圆形投影）
                    collision = False
                    for x_exist, y_exist, r_exist in placed:
                        dx = x_global - x_exist
                        dy = y_global - y_exist
                        if dy < effective_radius + r_exist:
                            collision = True
                            break

                    if not collision:
                        # 创建物体
                        h = random.random()  # 色相范围：0.0-1.0（对应0-360°）
                        s = random.uniform(0.3, 1)  # 饱和度范围：0.3-1.0（避免灰阶）
                        v = random.uniform(0.3, 0.8)  # 亮度范围：0.5-1.0（避免深色）
                        r, g, b = colorsys.hsv_to_rgb(h, s, v)
                        color = (
                            r,
                            g,
                            b,
                            1.0,
                        )
                        if shape_type == "cube":
                            vertices, normals = create_cuboid(
                                -half_size,
                                -half_size,
                                -half_size,
                                half_size,
                                half_size,
                                half_size,
                            )
                        else:
                            vertices, normals = create_sphere(half_size)

                        origin = np.eye(4)
                        origin[:3, 3] = [x_global, y_global, z_global]
                        self.objects.append(
                            Visual(origin, vertices, normals, color=color)
                        )
                        placed.append((x_global, y_global, effective_radius))
                        obj_placed = True
                        break  # 成功放置，跳出尝试循环

                if not obj_placed:
                    pass  # 经过多次尝试仍无法放置，跳过该物体

    def draw(self):
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.6, 0.6, 0.6, 1.0))
        self.visual.draw()
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))  # 恢复原材质
        for obj in self.objects:
            obj.draw()
