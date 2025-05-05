from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
from utils import parse_urdf, draw_link
import numpy as np
from shelf import create_shelf, draw_shelf

mouse_left_pressed = False
mouse_right_pressed = False
last_mouse_x, last_mouse_y = 0, 0
cam_radius = 5.196  # 初始距离 3√3 ≈ 5.196
cam_theta = np.pi / 4  # 初始水平角 (3,3,3)
cam_phi = np.arccos(3 / 5.196)  # 初始俯仰角
pan_offset = np.array([0.0, 0.0, 0.0])  # 存储场景平移量


def mouse_button_callback(window, button, action, mods):
    global mouse_left_pressed, mouse_right_pressed, last_mouse_x, last_mouse_y
    if button == glfw.MOUSE_BUTTON_LEFT:
        mouse_left_pressed = action == glfw.PRESS
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        mouse_right_pressed = action == glfw.PRESS
    else:
        return
    if action == glfw.PRESS:
        last_mouse_x, last_mouse_y = glfw.get_cursor_pos(window)


def cursor_pos_callback(window, xpos, ypos):
    global mouse_left_pressed, last_mouse_x, last_mouse_y, cam_theta, cam_phi, pan_offset
    delta_x = xpos - last_mouse_x
    delta_y = ypos - last_mouse_y
    last_mouse_x, last_mouse_y = xpos, ypos
    if mouse_left_pressed:
        # 调整视角角度
        sensitivity = 0.005
        cam_theta -= delta_x * sensitivity
        cam_phi -= delta_y * sensitivity  # 减号保证拖动方向符合直觉

        # 限制俯仰角范围
        cam_phi = np.clip(cam_phi, 0.1, np.pi - 0.1)
    elif mouse_right_pressed:
        # 计算摄像机方向向量
        x = np.sin(cam_phi) * np.cos(cam_theta)
        y = np.sin(cam_phi) * np.sin(cam_theta)
        z = np.cos(cam_phi)
        forward = np.array([x, y, z])
        right = np.cross(forward, [0, 0, 1])  # 计算右向量
        up = np.cross(right, forward)  # 计算上向量

        # 标准化向量
        right /= np.linalg.norm(right)
        up /= np.linalg.norm(up)

        # 计算平移量（速度与观察距离成正比）
        pan_speed = cam_radius * 0.002
        pan_offset += right * delta_x * pan_speed  # 水平移动
        pan_offset += up * delta_y * pan_speed  # 垂直移动


def scroll_callback(window, xoffset, yoffset):
    global cam_radius
    cam_radius -= yoffset * 0.5  # 调整缩放速度
    cam_radius = np.clip(cam_radius, 1.0, 20.0)  # 限制缩放范围


def key_callback(window, key, scancode, action, mods):
    global revolute_joints
    if action == glfw.PRESS or action == glfw.REPEAT:
        delta_angle = np.radians(5)  # 每次按键旋转5度

        # 映射数字键1~6到关节索引0~5
        if glfw.KEY_1 <= key <= glfw.KEY_6:
            index = key - glfw.KEY_1
            if index < len(revolute_joints):
                revolute_joints[index].angle += delta_angle
        # 增加反向旋转控制（小键盘1~6）
        elif key in [
            glfw.KEY_KP_1,
            glfw.KEY_KP_2,
            glfw.KEY_KP_3,
            glfw.KEY_KP_4,
            glfw.KEY_KP_5,
            glfw.KEY_KP_6,
        ]:
            index = key - glfw.KEY_KP_1
            if index < len(revolute_joints):
                revolute_joints[index].angle -= delta_angle


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
    glfw.set_key_callback(window, key_callback)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glMaterialfv(GL_FRONT, GL_SPECULAR, (0.5, 0.5, 0.5, 1))
    glMaterialfv(GL_FRONT, GL_SHININESS, 50)

    global revolute_joints
    root_link, revolute_joints = parse_urdf("robot/rm_65.urdf")
    # 创建货架
    shelf = create_shelf([0.2, 0.5, 0.8, 3], [0.5, 0.0, 0.0])

    while not glfw.window_should_close(window):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        width, height = glfw.get_framebuffer_size(window)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, width / height, 0.1, 100)
        x = pan_offset[0] + cam_radius * np.sin(cam_phi) * np.cos(cam_theta)
        y = pan_offset[1] + cam_radius * np.sin(cam_phi) * np.sin(cam_theta)
        z = pan_offset[2] + cam_radius * np.cos(cam_phi)

        # 设置模型视图矩阵
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            x,
            y,
            z,
            pan_offset[0],
            pan_offset[1],
            pan_offset[2],  # 看向平移后的目标点
            0,
            0,
            1,
        )

        draw_link(root_link)
        # 绘制货架并设置材质
        draw_shelf(shelf)
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
