import cv2
import glfw
import numpy as np
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_DIFFUSE,
    GL_FRONT,
    GL_LIGHT0,
    GL_LIGHTING,
    GL_LINE_LOOP,
    GL_MODELVIEW,
    GL_POSITION,
    GL_PROJECTION,
    GL_RGB,
    GL_SHININESS,
    GL_SPECULAR,
    GL_UNSIGNED_BYTE,
    glBegin,
    glClear,
    glClearColor,
    glColor3f,
    glDisable,
    glEnable,
    glEnd,
    glLightfv,
    glLineWidth,
    glLoadIdentity,
    glMaterialfv,
    glMatrixMode,
    glOrtho,
    glReadPixels,
    glVertex2f,
)
from OpenGL.GLU import gluLookAt, gluPerspective
from scipy.spatial.transform import Rotation as R

from animate import AnimationController
from cv import detect_objects
from robot import Robot
from shelf import Shelf
from utils import Floor, Target, parse_urdf

mouse_left_pressed = False
mouse_right_pressed = False
last_mouse_x, last_mouse_y = 0, 0
cam_radius = 5.196  # 初始距离 3√3 ≈ 5.196
cam_theta = np.pi / 4  # 初始水平角 (3,3,3)
cam_phi = np.arccos(3 / 5.196)  # 初始俯仰角
pan_offset = [0.0, 0.0, 0.0]  # 存储场景平移量
target_position = np.array([0.5, 0, 0.5])  # 初始目标位置
target_orientation = np.eye(3)  # 初始目标旋转


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
    global revolute_joints, target_orientation, target_position, objs
    target_axis = objs["target_axis"]
    robot = objs["robot"]
    if action != glfw.PRESS and action != glfw.REPEAT:
        return

    delta_angle = np.radians(5)  # 关节旋转步长
    delta_pos = 0.02  # 位置移动步长
    delta_rot = 5  # 旋转角度步长（度）

    # 获取Ctrl键状态
    ctrl_pressed = (
        glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
        or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
    )

    # 处理关节控制 1~7
    if glfw.KEY_1 <= key <= glfw.KEY_7:
        index = key - glfw.KEY_1
        # 根据Ctrl状态决定方向
        direction = -1 if ctrl_pressed else 1
        robot.move_joint(index, delta_angle * direction, True)
        target_position, target_orientation = robot.update_pos()
        target_axis.update_pose(target_position, target_orientation)

    # 处理位置控制 QWE (XYZ轴移动)
    pos_axes = {glfw.KEY_Q: 0, glfw.KEY_W: 1, glfw.KEY_E: 2}
    if key in pos_axes:
        axis = pos_axes[key]
        direction = -1 if ctrl_pressed else 1
        target_position[axis] += delta_pos * direction

    # 处理旋转控制 ASD (XYZ轴旋转)
    rot_axes = {glfw.KEY_A: "x", glfw.KEY_S: "y", glfw.KEY_D: "z"}
    if key in rot_axes:
        axis = rot_axes[key]
        # 根据Ctrl状态决定旋转方向
        angle = delta_rot * (-1 if ctrl_pressed else 1)
        rot = R.from_euler(axis, angle, degrees=True)
        target_orientation = rot.apply(target_orientation)

    # 触发逆解计算
    if key in [
        glfw.KEY_Q,
        glfw.KEY_W,
        glfw.KEY_E,
        glfw.KEY_A,
        glfw.KEY_S,
        glfw.KEY_D,
    ]:
        robot.update_joint_angles(target_position, target_orientation)
        target_axis.update_pose(target_position, target_orientation)

    if key == glfw.KEY_G and action == glfw.PRESS:
        anim_controller = objs["anim_controller"]
        try:
            anim_controller.start_pick_and_place()
        except Exception as e:
            print(f"抓取失败: {str(e)}")
            anim_controller._reset_state()


def gl_init(window):
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


def gl_init_shelf(window):
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (1, 1, 1, 0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (1, 1, 1, 1))
    glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.8, 0.8, 0.8, 1))
    glMaterialfv(GL_FRONT, GL_SPECULAR, (0.5, 0.5, 0.5, 1))
    glMaterialfv(GL_FRONT, GL_SHININESS, 50)


def render_main_window(window, objs):
    glClearColor(10 / 255, 204 / 255, 255 / 255, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glClear(GL_DEPTH_BUFFER_BIT)
    width, height = glfw.get_framebuffer_size(window)

    # 设置投影矩阵
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100)

    # 计算摄像机位置
    x = pan_offset[0] + cam_radius * np.sin(cam_phi) * np.cos(cam_theta)
    y = pan_offset[1] + cam_radius * np.sin(cam_phi) * np.sin(cam_theta)
    z = pan_offset[2] + cam_radius * np.cos(cam_phi)

    # 设置模型视图矩阵
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(x, y, z, pan_offset[0], pan_offset[1], pan_offset[2], 0, 0, 1)

    # 绘制场景
    objs["root_link"].draw()
    objs["shelf"].draw()
    objs["target_axis"].draw()
    objs["floor"].draw()
    objs["anim_controller"].draw_obj()
    glfw.swap_buffers(window)


def render_shelf_window(window, objs):
    shelf = objs["shelf"]
    glClearColor(102 / 255, 204 / 255, 255 / 255, 1.0)
    glClear(GL_COLOR_BUFFER_BIT)
    glClear(GL_DEPTH_BUFFER_BIT)
    width, height = glfw.get_framebuffer_size(window)

    # 设置投影矩阵
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / height, 0.1, 100)

    # 设置固定摄像机视角（货架后方）
    shelf_pos = shelf.position
    camera_distance = 1.0  # 观察距离
    camera_height = shelf.box_size[2] / 2  # 摄像机高度偏移

    # 摄像机位置（货架后方，Y轴负方向）
    cam_x = shelf_pos[0] + camera_distance
    cam_y = shelf_pos[1]
    cam_z = shelf_pos[2] + camera_height

    # 设置模型视图矩阵
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(
        cam_x,
        cam_y,
        cam_z,  # 摄像机位置
        shelf_pos[0],
        shelf_pos[1],
        cam_z,  # 观察点（货架中心）
        0,
        0,
        1,
    )  # 上方向

    # 绘制场景
    objs["root_link"].draw()
    shelf.draw()
    objs["floor"].draw()
    # 禁用深度测试和光照以绘制2D元素
    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    # 新增：读取当前帧并检测物体
    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
    assert isinstance(data, bytes)
    image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    image = cv2.flip(image, 0)  # 垂直翻转图像
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 执行物体检测
    detections = detect_objects(image)

    # 设置2D正交投影
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, width, 0, height, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glLineWidth(3.0)  # 将线宽从默认1.0增加到3.0
    glColor3f(0, 1, 0)  # 设置绿色
    # 绘制检测框（转换为OpenGL坐标系）
    for x, y, w, h in detections:
        y_gl = height - y - h  # 转换Y坐标
        glBegin(GL_LINE_LOOP)
        glVertex2f(x, y_gl)
        glVertex2f(x + w, y_gl)
        glVertex2f(x + w, y_gl + h)
        glVertex2f(x, y_gl + h)
        glEnd()

    # 恢复3D渲染状态
    glLineWidth(1.0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glfw.swap_buffers(window)


def main():
    assert glfw.init() == True
    # 创建主窗口
    glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)  # 确保启用双缓冲
    glfw.window_hint(glfw.SAMPLES, 4)  # 开启4x多重采样抗锯齿
    window = glfw.create_window(800, 600, "Robot Viewer", None, None)
    if not window:
        glfw.terminate()
        return
    gl_init(window)
    # 创建货架观察窗口（共享上下文）
    shelf_window = glfw.create_window(800, 600, "Shelf View", None, window)
    if not shelf_window:
        glfw.terminate()
        return
    gl_init_shelf(shelf_window)
    global revolute_joints, objs
    root_link, revolute_joints = parse_urdf("robot/rm_65.urdf")
    objs = {
        "root_link": root_link,
        "robot": Robot("robot/rm_65.urdf", "robot/arm.urdf", revolute_joints),
        "shelf": Shelf([0.15, 0.5, 0.5, 3], [0.3, 0.0, 0.3]),
        "target_axis": Target(),
        "floor": Floor(size=2, step=0.5),
    }
    anim_controller = AnimationController(
        robot=objs["robot"],
        shelf=objs["shelf"],
        receive_position=np.array([-0.5, 0, 0.5]),
    )
    objs["anim_controller"] = anim_controller

    while not (
        glfw.window_should_close(window) or glfw.window_should_close(shelf_window)
    ):
        if anim_controller.is_animating:
            anim_controller.update()
        # 渲染主窗口
        glfw.make_context_current(window)
        render_main_window(window, objs)
        # 渲染货架观察窗口
        glfw.make_context_current(shelf_window)
        render_shelf_window(shelf_window, objs)

        glfw.poll_events()
    glfw.terminate()


if __name__ == "__main__":
    main()
