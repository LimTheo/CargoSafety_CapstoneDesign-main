import cv2
import numpy as np
from PIL import Image
from math import ceil

def detect_pallet_tilt(image_input, mean_threshold=3.0, std_threshold=2.0):
    """
    실시간용 빠른 기울기 계산 함수 (그래프 없음).
    """
    if isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    else:
        image = image_input.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=20
    )

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)

            if dy == 0 or abs(dx) > abs(dy):
                continue

            angle_rad = np.arctan(dx / dy)
            angle_deg = abs(np.degrees(angle_rad))

            if angle_deg < 45:
                angles.append(angle_deg)

    if len(angles) == 0:
        return "NORMAL (no lines)", 0.0, 0.0

    mean_angle = np.mean(angles)
    std_angle = np.std(angles)

    if mean_angle > mean_threshold:
        status = "WARNING: TILTED"
    elif std_angle > std_threshold:
        status = "WARNING: UNSTABLE"
    else:
        status = "NORMAL"

    return status, mean_angle, std_angle
