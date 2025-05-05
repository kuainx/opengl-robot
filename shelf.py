import numpy as np
from OpenGL.GL import GL_DIFFUSE, GL_FRONT, glMaterialfv

from utils import Visual, create_cuboid


class Shelf:
    def __init__(self, box_size, position):
        self.box_size = box_size
        shelf_vertices, shelf_normals = self.create_visual()
        shelf_origin = np.eye(4)
        shelf_origin[:3, 3] = position  # 调整货架位置
        self.visual = Visual(shelf_origin, shelf_vertices, shelf_normals)

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

    def draw(self):
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.6, 0.6, 0.6, 1.0))
        self.visual.draw()
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1.0))  # 恢复原材质
