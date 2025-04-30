from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
from utils import parse_urdf
import numpy as np

mouse_left_pressed = False
last_mouse_x, last_mouse_y = 0, 0
cam_radius = 5.196  # 初始距离 3√3 ≈ 5.196
cam_theta = np.pi / 4  # 初始水平角 (3,3,3)
cam_phi = np.arccos(3 / 5.196)  # 初始俯仰角


def draw_visual(visual):
    glPushMatrix()
    glMultMatrixf(visual.origin.T.ravel())  # OpenGL需要列主序
    vertices = visual.vertices
    normals = visual.normals
    glBegin(GL_TRIANGLES)
    for v, n in zip(vertices, normals):
        glNormal3fv(n)
        glVertex3fv(v)
    glEnd()
    glPopMatrix()


def draw_link(link):
    for visual in link.visuals:
        draw_visual(visual)
    for joint in link.children:
        glPushMatrix()
        glMultMatrixf(joint.origin.T.ravel())
        draw_link(joint.child)
        glPopMatrix()


def mouse_button_callback(window, button, action, mods):
    global mouse_left_pressed, last_mouse_x, last_mouse_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            mouse_left_pressed = True
            last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            mouse_left_pressed = False


def cursor_pos_callback(window, xpos, ypos):
    global mouse_left_pressed, last_mouse_x, last_mouse_y, cam_theta, cam_phi
    if mouse_left_pressed:
        delta_x = xpos - last_mouse_x
        delta_y = ypos - last_mouse_y
        last_mouse_x, last_mouse_y = xpos, ypos

        # 调整视角角度
        sensitivity = 0.005
        cam_theta += delta_x * sensitivity
        cam_phi -= delta_y * sensitivity  # 减号保证拖动方向符合直觉

        # 限制俯仰角范围
        cam_phi = np.clip(cam_phi, 0.1, np.pi - 0.1)


def scroll_callback(window, xoffset, yoffset):
    global cam_radius
    cam_radius -= yoffset * 0.5  # 调整缩放速度
    cam_radius = np.clip(cam_radius, 1.0, 20.0)  # 限制缩放范围


def main():
    if not glfw.init():
        return
    window = glfw.create_window(800, 600, "URDF Viewer", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glMaterialfv(GL_FRONT, GL_SPECULAR, (0.5, 0.5, 0.5, 1))
    glMaterialfv(GL_FRONT, GL_SHININESS, 50)

    root_link = parse_urdf("robot/rm_65.urdf")  # 替换为你的URDF文件路径

    while not glfw.window_should_close(window):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100)
        x = cam_radius * np.sin(cam_phi) * np.cos(cam_theta)
        y = cam_radius * np.sin(cam_phi) * np.sin(cam_theta)
        z = cam_radius * np.cos(cam_phi)

        # 设置模型视图矩阵
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(x, y, z, 0, 0, 0, 0, 0, 1)  # 保持up方向为z轴

        draw_link(root_link)
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
