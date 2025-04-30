import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
import struct
import os


def parse_stl(filename):
    """解析STL文件，支持ASCII和二进制格式，返回顶点和法线列表"""
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


# 交互控制变量
mouse_down = False
last_x, last_y = 0, 0
rotate_x, rotate_y = 0, 0
zoom = 0.5


def main():
    global rotate_x, rotate_y, zoom

    # 初始化GLFW
    if not glfw.init():
        return

    # 创建窗口
    window = glfw.create_window(800, 600, "STL Viewer", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_window_size_callback(window, on_window_size)

    # 配置OpenGL
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, (0.7, 0.7, 0.7, 1.0))

    # 加载模型
    vertices, normals = parse_stl("./robot/meshes/link1.stl")  # 修改为你的文件路径

    # 主循环
    while not glfw.window_should_close(window):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        # glColor3f(1, 0, 0)
        glColor3f(1.0, 1.0, 1.0)
        glLoadIdentity()
        gluLookAt(0, 0, zoom, 0, 0, 0, 0, 1, 0)
        glRotatef(rotate_x, 1, 0, 0)
        glRotatef(rotate_y, 0, 1, 0)

        # 绘制模型
        glBegin(GL_TRIANGLES)
        for v, n in zip(vertices, normals):
            print(n)
            glNormal3fv(n)
            glVertex3fv(v)
        glEnd()

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


# 回调函数
def on_mouse_button(window, button, action, mods):
    global mouse_down, last_x, last_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_down = action == glfw.PRESS
        last_x, last_y = glfw.get_cursor_pos(window)


def on_cursor_pos(window, x, y):
    global last_x, last_y, rotate_x, rotate_y
    if mouse_down:
        dx, dy = x - last_x, y - last_y
        rotate_y += dx * 0.5
        rotate_x += dy * 0.5
        last_x, last_y = x, y


def on_scroll(window, dx, dy):
    global zoom
    zoom += dy * 0.5
    print(zoom)


def on_window_size(window, width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)


if __name__ == "__main__":
    main()
