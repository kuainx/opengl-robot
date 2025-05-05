import numpy as np
from utils import Visual, draw_visual
from OpenGL.GL import *


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


def create_shelf_visual(width, depth, height, num_shelves):
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
        pillar_verts, pillar_norms = create_cuboid(x1, y1, z, x2, y2, z + height)
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


def create_shelf(box_size, position):
    shelf_vertices, shelf_normals = create_shelf_visual(
        box_size[0], box_size[1], box_size[2], box_size[3]
    )
    shelf_origin = np.eye(4)
    shelf_origin[:3, 3] = position  # 调整货架位置
    return Visual(shelf_origin, shelf_vertices, shelf_normals)


def draw_shelf(shelf):
    glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.6, 0.6, 0.6, 1.0))
    draw_visual(shelf)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))  # 恢复原材质
