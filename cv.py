import cv2
import numpy as np


def detect_objects(image):
    """使用OpenCV检测货架物体"""
    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定义物体颜色范围（示例检测红色物体）
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([255, 255, 200])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 形态学操作去噪
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:  # 过滤小噪点
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))

    return detections
