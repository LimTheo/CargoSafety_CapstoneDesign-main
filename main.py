# import cv2
# import os
# from picamera2 import Picamera2
# from huggingface_hub import hf_hub_download
# from PIL import Image, ImageDraw
# from src.detection.object_detection import detect_and_crop
# from src.detection.masking import mask_background
# from src.tilt.tilt_detection import detect_pallet_tilt_with_graph
# from src.common.camera_input import get_camera_stream
# from src.common.visualization import show_image, preview_dataset

# # 데이터 다운로드 (처음 실행 시)
# if not os.path.exists('data/yoloe-v8l-seg.pt'):
#     hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg.pt", local_dir='data')
#     hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg-pf.pt", local_dir='data')
# if not os.path.exists('data/ram_tag_list.txt'):
#     os.system('wget https://raw.githubusercontent.com/THU-MIG/yoloe/main/tools/ram_tag_list.txt -P data')
# # 다른 wget 파일도 유사하게 (bus.jpg 등, 테스트용)

# for frame in get_camera_stream():
#     # 1. 화물 검출 & 크롭
#     detection_result = detect_and_crop(frame)
#     boxes = detection_result["boxes"]
#     dataset = detection_result["dataset"]
#     annotated_img = Image.fromarray(detection_result["annotated_image"])

#     # 2. 마스킹 (선택적)
#     masked_img = mask_background(annotated_img, boxes)

#     # 3. 각 크롭에 기울기 계산
#     for item in dataset:
#         print(f"Analyzing cropped image for: {item['label']}")
#         result_img, status, mean, std = detect_pallet_tilt_with_graph(item['image'])
#         if result_img is not None:
#             print(f"[{status}] Mean: {mean:.2f}, Std: {std:.2f}")
#             show_image(result_img, f"Tilt Result for {item['label']}", wait=False)

#     # 결과 표시
#     show_image(annotated_img, 'Detected Boxes', wait=False)
#     show_image(masked_img, 'Masked Image', wait=False)
#     # preview_dataset(dataset)  # 필요시 (matplotlib, 실시간에 느림)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
#         break

# cv2.destroyAllWindows()




# # new code
# import cv2
# from src.detection.object_detection import detect_and_crop
# from src.tilt.tilt_detection import detect_pallet_tilt
# from src.common.camera_input import get_camera_stream

# frame_count = 0
# last_detection = None

# for frame in get_camera_stream():

#     frame_count += 1

#     # -----------------------------------------
#     # 1) YOLO는 3프레임마다 한 번만 실행 (속도 향상)
#     # -----------------------------------------
#     if frame_count % 3 == 0:
#         last_detection = detect_and_crop(frame)

#     if last_detection:
#         boxes = last_detection["boxes"]
#         dataset = last_detection["dataset"]

#         # 박스 Draw
#         for box in boxes:
#             x1, y1, x2, y2 = box.astype(int)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

#         # 기울기 계산
#         if len(dataset) > 0:
#             crop_img = dataset[0]["image"]
#             status, mean, std = detect_pallet_tilt(crop_img)

#             color = (0,255,0) if "NORMAL" in status else (0,0,255)
#             cv2.putText(frame, f"{status} | Mean={mean:.2f}, Std={std:.2f}",
#                         (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         1.0, color, 2)

#     # -----------------------------------------
#     # 2) 실시간 영상 출력 (단 하나의 창)
#     # -----------------------------------------
#     cv2.imshow("Real-Time Cargo Tilt Detection", frame)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break

# cv2.destroyAllWindows()





# # main.py  -- Raspberry Pi friendly, contour-based box detection + tilt check
# import time
# import threading
# import cv2
# import numpy as np
# from picamera2 import Picamera2
# from collections import deque

# # --- 설정 (환경에 맞게 조절) ---
# CAM_SIZE = (1280, 720)
# CAM_BUFFER = 2
# MOTION_ALPHA = 0.04
# MOTION_THRESHOLD = 20
# MOTION_MIN_AREA = 1500
# MOTION_HISTORY = 4
# TILT_MEAN_TH = 3.0
# TILT_STD_TH = 2.0
# DISPLAY_WINDOW = "Cargo Live (Contour -> Tilt)"
# RUNNING = True

# # --- 전역 프레임 저장 (thread-safe) ---
# _latest_frame = None
# _frame_lock = threading.Lock()

# # --------------------
# # 카메라 스레드
# # --------------------
# def camera_worker():
#     global _latest_frame, RUNNING
#     picam2 = Picamera2()
#     config = picam2.create_video_configuration(
#         main={"size": CAM_SIZE, "format": "RGB888"},
#         buffer_count=CAM_BUFFER
#     )
#     picam2.configure(config)
#     picam2.start()
#     time.sleep(0.2)
#     print("[CAM] started (RGB888)")

#     while RUNNING:
#         try:
#             frame = picam2.capture_array("main")
#             if frame is None:
#                 continue
#             # RGB -> BGR for OpenCV
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#             with _frame_lock:
#                 _latest_frame = frame
#         except Exception as e:
#             print("[CAM ERR]", e)
#             time.sleep(0.1)
#     try:
#         picam2.stop()
#     except Exception:
#         pass

# # --------------------
# # Motion Detector (background average)
# # --------------------
# class MotionDetector:
#     def __init__(self, alpha=MOTION_ALPHA):
#         self.bg = None
#         self.alpha = alpha
#         self.history = deque(maxlen=MOTION_HISTORY)

#     def apply(self, frame_bgr):
#         gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (7,7), 0)
#         if self.bg is None:
#             self.bg = gray.astype("float32")
#             self.history.append(False)
#             return np.zeros_like(gray), False
#         cv2.accumulateWeighted(gray.astype("float32"), self.bg, self.alpha)
#         bg_uint8 = cv2.convertScaleAbs(self.bg)
#         diff = cv2.absdiff(gray, bg_uint8)
#         _, th = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
#         th = cv2.medianBlur(th, 5)
#         th = cv2.dilate(th, None, iterations=1)
#         # largest contour area
#         contours, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         max_area = 0
#         for c in contours:
#             a = cv2.contourArea(c)
#             if a > max_area:
#                 max_area = a
#         flag = max_area > MOTION_MIN_AREA
#         self.history.append(flag)
#         motion = sum(self.history) > (len(self.history)//2)
#         return th, motion

# # --------------------
# # Contour-based box detector (fast)
# # Returns list of boxes [x1,y1,x2,y2]
# # --------------------
# def detect_boxes_by_contour(frame_bgr, min_area=2000):
#     gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#     _, th = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     # morphological close -> remove small holes
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
#     th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
#     # find contours
#     contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     boxes = []
#     for c in contours:
#         area = cv2.contourArea(c)
#         if area < min_area:
#             continue
#         x,y,w,h = cv2.boundingRect(c)
#         # aspect ratio / size heuristic (필요시 조정)
#         if w < 30 or h < 30:
#             continue
#         boxes.append([x, y, x+w, y+h])
#     return boxes, th

# # --------------------
# # Fast tilt detector (use existing Hough approach but lighter)
# # Input: crop (BGR)
# # Output: status, mean, std
# # --------------------
# def detect_pallet_tilt_fast(crop_bgr, mean_threshold=TILT_MEAN_TH, std_threshold=TILT_STD_TH):
#     if crop_bgr is None or crop_bgr.size == 0:
#         return "NO_DATA", 0.0, 0.0
#     img = crop_bgr.copy()
#     h, w = img.shape[:2]
#     if h > 400:
#         scale = 400 / h
#         img = cv2.resize(img, (int(w*scale), int(h*scale)))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     edges = cv2.Canny(blur, 50, 150)
#     lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, minLineLength=max(8, img.shape[0]//10), maxLineGap=8)
#     angles = []
#     if lines is not None:
#         for l in lines:
#             x1,y1,x2,y2 = l[0]
#             dx = float(x2-x1); dy = float(y2-y1)
#             if dy == 0 or abs(dx) > abs(dy): continue
#             ang = abs(np.degrees(np.arctan2(dx, dy)))
#             if ang <= 45:
#                 angles.append(ang)
#     if not angles:
#         return "NORMAL", 0.0, 0.0
#     arr = np.array(angles)
#     mean_ang = float(arr.mean()); std_ang = float(arr.std())
#     if mean_ang > mean_threshold:
#         return "TILTED", mean_ang, std_ang
#     if std_ang > std_threshold:
#         return "UNSTABLE", mean_ang, std_ang
#     return "NORMAL", mean_ang, std_ang

# # --------------------
# # (옵션) 서버 오프로드 placeholder
# # --------------------
# def send_crop_to_server_and_get_result(crop_bgr):
#     """
#     OPTIONAL: 구현하면 좋음.
#     - crop_bgr: numpy BGR
#     - 여기서 HTTP로 서버에 업로드 → YOLOE(또는 MobileCLIP)로 추론 → 결과 반환
#     현재는 미구현(디버그용).
#     """
#     return None

# # --------------------
# # MAIN LOOP
# # --------------------
# def main_loop():
#     global RUNNING
#     cam_thread = threading.Thread(target=camera_worker, daemon=True)
#     cam_thread.start()

#     md = MotionDetector()
#     last_infer = 0
#     COOLDOWN = 1.0  # 초 단위: 연속 추론 방지

#     cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)

#     try:
#         while True:
#             with _frame_lock:
#                 frame = None if _latest_frame is None else _latest_frame.copy()
#             if frame is None:
#                 time.sleep(0.01)
#                 continue

#             h, w = frame.shape[:2]

#             # motion detection (light)
#             mask, motion = md.apply(frame)
#             # small mask preview on top-right
#             try:
#                 m_small = cv2.resize(mask, (160,120))
#                 frame[10:10+120, w-10-160:w-10] = cv2.cvtColor(m_small, cv2.COLOR_GRAY2BGR)
#             except:
#                 pass

#             # detect boxes by contour (very fast)
#             boxes, contour_mask = detect_boxes_by_contour(frame, min_area=MOTION_MIN_AREA)

#             # If boxes found AND cooldown passed -> compute tilt for each
#             now = time.time()
#             if boxes and (now - last_infer) > COOLDOWN:
#                 last_infer = now
#                 # (옵션) 서버 전송 부분: for each crop you could call send_crop_to_server...
#                 # 여기서는 로컬 빠른 기울기만 수행
#                 for i, box in enumerate(boxes):
#                     x1,y1,x2,y2 = box
#                     crop = frame[y1:y2, x1:x2]
#                     status, mean, std = detect_pallet_tilt_fast(crop)
#                     color = (0,255,0) if status == "NORMAL" else (0,165,255) if status == "UNSTABLE" else (0,0,255)
#                     label = f"{status} M={mean:.1f} S={std:.1f}"
#                     cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#                     cv2.putText(frame, label, (x1, max(10, y2+16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#             # show motion + count
#             cv2.putText(frame, f"Motion:{int(motion)} Boxes:{len(boxes)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

#             cv2.imshow(DISPLAY_WINDOW, frame)
#             key = cv2.waitKey(1) & 0xFF
#             if key == 27 or key == ord('q'):
#                 break

#     finally:
#         RUNNING = False
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main_loop()


# main.py
# YOLOE (yoloe-v8l-seg.pt) 기반 실시간 카메라 + 변화감지 트리거 + 기존 기울기 함수 통합
# Raspberry Pi 용으로 모션 기반으로만 무거운 추론을 수행하도록 설계


import time
import threading
import io
import math
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw
from picamera2 import Picamera2

# YOLOE import (THU-MIG yoloe wrapper)
from ultralytics import YOLOE

# -----------------------
# 설정
# -----------------------
MODEL_PATH = "/home/pi/Desktop/CargoSafety_CapstoneDesign-main/yoloe-v8s-seg.pt"   # 네가 가진 모델 파일 경로 (필요하면 수정)
DEVICE = "cpu"                    # raspbePrry pi: "cpu"
CAM_SIZE = (1280, 720)            # 16:9 (너가 원함)
CAM_BUFFER = 4
MOTION_ALPHA = 0.03
MOTION_THRESHOLD = 20
MOTION_MIN_AREA = 1500
MOTION_HISTORY = 5
YOLO_IMG_SIZE = 640               # 추론 입력 크기 (작게 하면 빠름)
YOLO_COOLDOWN = 2.5               # 초: 마지막 추론 후 대기
DISPLAY_WINDOW = "Cargo YOLOE Live"

# 네가 쓰던 클래스 리스트 (원본)
NAMES = [
    "cardboard_box_front", "cardboard_box_diagonal", "cardboard_box_tilted",
    "cardboard_box_heavily_tilted", "cardboard_box_stacked", "cardboard_box_collapsed",
    "cardboard_box_damaged",
    "plastic_container_front", "plastic_container_tilted", "plastic_container_stacked",
    "plastic_container_damaged",
    "metal_case_front", "metal_case_tilted", "metal_case_damaged",
    "wooden_crate_front", "wooden_crate_tilted", "wooden_crate_damaged",
    "stacked_boxes", "leaning_box", "displaced_box", "collapsed_box",
    "wrapped_cargo", "open_cargo", "pallet_wrapped", "pallet_open",
    "other_cargo"
]

# -----------------------
# 전역 상태 (스레드 안전)
# -----------------------
_latest_frame = None
_frame_lock = threading.Lock()
running = True

# YOLOE 상태
_yolo_model = None
_yolo_lock = threading.Lock()
_yolo_running = False
_last_yolo_time = 0.0
_latest_detection = None
_detection_lock = threading.Lock()


# -----------------------
# 유틸: 히스토그램을 이미지로 (너가 쓰던 그대로)
# -----------------------
def draw_histogram_to_image(data_list, mean_val, std_val, height):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(data_list, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    ax.axvspan(mean_val - std_val, mean_val + std_val, color='green', alpha=0.1, label=f'Std: {std_val:.2f}')
    ax.set_title('Tilt Angle Distribution')
    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel('Count')
    ax.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plot_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close(fig)
    h_plot, w_plot = plot_img.shape[:2]
    scale = height / h_plot
    plot_img_resized = cv2.resize(plot_img, (int(w_plot * scale), height))
    return plot_img_resized


# -----------------------
# 기존의 detect_pallet_tilt_with_graph (원형 그대로 재현)
# 입력: PIL.Image 또는 path 문자열
# 반환: final_result(BGR np.ndarray), status, avg, std
# -----------------------
def detect_pallet_tilt_with_graph(image_input, mean_threshold=3.0, std_threshold=2.0):
    """
    그대로 너가 전에 쓰던 함수 로직을 유지. 입력은 PIL 이미지 또는 파일경로.
    최종적으로 OpenCV(BGR) 이미지를 리턴.
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        image = np.array(image_input)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        # assume BGR or RGB? we assume BGR (we'll pass BGR crops)
        image = image_input.copy()
    else:
        return None, "Error: Invalid image input type", 0, 0

    if image is None:
        return None, "Error: Image not found or could not be loaded", 0, 0

    # 리사이즈 (높이 800 고정)
    target_height = 600  # crop이 작으니 높이 줄임 (성능)
    h, w = image.shape[:2]
    if h == 0:
        return None, "Error: Empty image", 0, 0
    scale = target_height / h
    image = cv2.resize(image, (int(w * scale), target_height))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=target_height / 10,
        maxLineGap=20
    )

    output_image = image.copy()
    angles = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            if dy == 0 or abs(dx) > abs(dy):
                continue
            angle_rad = np.arctan(dx / dy)
            angle_deg = math.ceil(np.abs(np.degrees(angle_rad)))
            if angle_deg > 45:
                continue
            vertical_lines.append(line)
            angles.append(angle_deg)

    if not angles:
        avg_angle = 0.0
        std_dev_angle = 0.0
        status = "NORMAL (No lines)"
        color = (0, 255, 0)
    else:
        angles_np = np.array(angles)
        avg_angle = np.mean(angles_np)
        std_dev_angle = np.std(angles_np)
        is_tilted = avg_angle > mean_threshold
        is_unstable = std_dev_angle > std_threshold
        if is_tilted:
            status = "WARNING: TILTED"
            color = (0, 0, 255)
        elif is_unstable:
            status = "WARNING: UNSTABLE"
            color = (0, 165, 255)
        else:
            status = "NORMAL"
            color = (0, 255, 0)
        for line in vertical_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output_image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    cv2.rectangle(output_image, (0, 0), (450, 140), (255, 255, 255), -1)
    cv2.putText(output_image, f"Status: {status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output_image, f"Mean Angle: {avg_angle:.2f} (Th: {mean_threshold})", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,50,50), 2)
    cv2.putText(output_image, f"Std Dev: {std_dev_angle:.2f} (Th: {std_threshold})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,50,50), 2)

    if angles:
        hist_img = draw_histogram_to_image(angles, avg_angle, std_dev_angle, target_height)
        final_result = np.hstack((output_image, hist_img))
    else:
        final_result = output_image

    return final_result, status, avg_angle, std_dev_angle


# -----------------------
# Camera worker: 항상 최신 프레임 갱신
# -----------------------
def camera_worker():
    global _latest_frame, running
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": CAM_SIZE, "format": "RGB888"},
        buffer_count=CAM_BUFFER
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(0.2)

    print("[CAM] started (RGB888)")

    while running:
        try:
            frame = picam2.capture_array("main")
            if frame is None:
                continue
            # RGB -> BGR for display and OpenCV processing
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                frame_bgr = frame.copy()

            with _frame_lock:
                _latest_frame = frame_bgr
        except Exception as e:
            print("[CAM ERROR]", e)
            time.sleep(0.1)

    try:
        picam2.stop()
    except Exception:
        pass


# -----------------------
# Motion detector (lightweight)
# -----------------------
class MotionDetector:
    def __init__(self, alpha=MOTION_ALPHA):
        self.bg = None
        self.alpha = alpha
        self.history = deque(maxlen=MOTION_HISTORY)

    def apply(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        if self.bg is None:
            self.bg = gray.astype("float32")
            self.history.append(False)
            return np.zeros_like(gray), False
        cv2.accumulateWeighted(gray.astype("float32"), self.bg, self.alpha)
        bg_uint8 = cv2.convertScaleAbs(self.bg)
        diff = cv2.absdiff(gray, bg_uint8)
        _, th = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        th = cv2.medianBlur(th, 5)
        th = cv2.dilate(th, None, iterations=1)
        contours, _ = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        for c in contours:
            a = cv2.contourArea(c)
            if a > max_area:
                max_area = a
        motion_flag = max_area > MOTION_MIN_AREA
        self.history.append(motion_flag)
        motion = sum(self.history) > (len(self.history) // 2)
        return th, motion


def masks_to_boxes(res, orig_w, orig_h, resized_w, resized_h, min_area=200):
    """
    YOLOE segmentation result -> list of boxes [x1,y1,x2,y2], scores, labels
    - res: single result object (results[0])
    - orig_w, orig_h: 원본 이미지 크기
    - resized_w, resized_h: 모델 입력으로 사용된 크기 (rescale factor used earlier)
    Returns: boxes, scores, labels
    """
    boxes = []
    scores = []
    labels = []

    # scale factors to map mask coords back to original
    sx = orig_w / float(resized_w)
    sy = orig_h / float(resized_h)

    # Try multiple common mask outputs that YOLOE variants might expose
    # 1) res.masks.xy  (polygons)
    try:
        if hasattr(res, "masks") and hasattr(res.masks, "xy") and res.masks.xy is not None:
            # res.masks.xy: list of polygons (list of list of (x,y) pairs) in resized coords
            for i, poly in enumerate(res.masks.xy):
                if poly is None: 
                    continue
                poly_pts = np.array(poly).reshape(-1, 2)
                if poly_pts.size == 0:
                    continue
                x1 = int(np.min(poly_pts[:,0] * sx))
                y1 = int(np.min(poly_pts[:,1] * sy))
                x2 = int(np.max(poly_pts[:,0] * sx))
                y2 = int(np.max(poly_pts[:,1] * sy))
                area = (x2-x1) * (y2-y1)
                if area < min_area: 
                    continue
                boxes.append([max(0,x1), max(0,y1), min(orig_w-1,x2), min(orig_h-1,y2)])
                # labels/scores may be stored separately
                try:
                    labels.append(int(res.boxes.cls[i].cpu().numpy()[0]))
                except Exception:
                    labels.append(None)
                try:
                    scores.append(float(res.boxes.conf[i].cpu().numpy()[0]))
                except Exception:
                    scores.append(0.0)
            if boxes:
                return boxes, scores, labels
    except Exception:
        pass

    # 2) res.masks.data or res.masks.data.cpu() -> binary masks (H x W x N) or flattened
    # We will try to extract mask bitmaps and find contours.
    try:
        if hasattr(res, "masks") and (hasattr(res.masks, "data") or hasattr(res.masks, "segmentation")):
            # Attempt to get per-mask binary arrays
            mask_arrays = None
            # different versions expose different fields
            if hasattr(res.masks, "data"):
                # sometimes data is a tensor (N,H,W) or a packed format; try numpy
                try:
                    mask_tensor = res.masks.data
                    # If it's torch tensor
                    import torch
                    if isinstance(mask_tensor, torch.Tensor):
                        mask_arrays = mask_tensor.cpu().numpy()
                    else:
                        mask_arrays = np.array(mask_tensor)
                except Exception:
                    mask_arrays = None
            elif hasattr(res.masks, "segmentation"):
                try:
                    mask_arrays = np.array(res.masks.segmentation)
                except Exception:
                    mask_arrays = None

            if mask_arrays is not None:
                # mask_arrays shape may be (N,H,W) or (H,W,N) or (H,W)
                if mask_arrays.ndim == 2:
                    mask_arrays = mask_arrays[np.newaxis, ...]
                if mask_arrays.ndim == 3:
                    # assume (N,H,W) or (H,W,N)
                    if mask_arrays.shape[0] != resized_h and mask_arrays.shape[-1] == resized_h:
                        # transpose if needed
                        mask_arrays = mask_arrays.transpose(2,0,1)
                    N = mask_arrays.shape[0]
                    for i in range(N):
                        m = (mask_arrays[i] > 0.5).astype(np.uint8) * 255
                        # resize mask to original scale if mask is in resized coords
                        if m.shape[0] != orig_h or m.shape[1] != orig_w:
                            m_big = cv2.resize(m, (resized_w, resized_h))  # ensure same resized size
                            m_big = cv2.resize(m_big, (orig_w, orig_h))
                        else:
                            m_big = m
                        contours, _ = cv2.findContours(m_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue
                        # take largest contour
                        c = max(contours, key=cv2.contourArea)
                        area = cv2.contourArea(c)
                        if area < min_area:
                            continue
                        x,y,w_box,h_box = cv2.boundingRect(c)
                        boxes.append([x,y,x+w_box,y+h_box])
                        # labels/scores unknown here
                        labels.append(None)
                        scores.append(0.0)
                    if boxes:
                        return boxes, scores, labels
    except Exception:
        pass

    # 3) res.masks.xy if stored as np.array form or res.masks.polygons
    try:
        # fallback checking for 'polys' attr
        if hasattr(res, "masks") and hasattr(res.masks, "polys"):
            for poly in res.masks.polys:
                poly_pts = np.array(poly).reshape(-1,2)
                x1 = int(np.min(poly_pts[:,0]*sx)); y1 = int(np.min(poly_pts[:,1]*sy))
                x2 = int(np.max(poly_pts[:,0]*sx)); y2 = int(np.max(poly_pts[:,1]*sy))
                area = (x2-x1)*(y2-y1)
                if area < min_area: continue
                boxes.append([x1,y1,x2,y2]); labels.append(None); scores.append(0.0)
            if boxes:
                return boxes, scores, labels
    except Exception:
        pass

    # last-resort: return empty
    return [], [], []

# -----------------------
# YOLOE 비동기 추론 스레드
# -----------------------
def run_yoloe_async(frame_bgr):
    global _yolo_model, _yolo_running, _last_yolo_time, _latest_detection

    def worker(img_bgr):
        global _yolo_model, _yolo_running, _last_yolo_time, _latest_detection
        try:
            with _yolo_lock:
                if _yolo_model is None:
                    print("[YOLOE] Loading model (this may take time)...")
                    _yolo_model = YOLOE(MODEL_PATH).to(DEVICE)
                    try:
                        _yolo_model.set_classes(NAMES, _yolo_model.get_text_pe(NAMES))
                    except Exception:
                        pass

            # prepare rgb resized
            orig_h, orig_w = img_bgr.shape[:2]
            new_w = YOLO_IMG_SIZE
            new_h = int(orig_h * (YOLO_IMG_SIZE / orig_w))
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb, (new_w, new_h))
            pil_img = Image.fromarray(rgb_resized)

            results = _yolo_model.predict(pil_img, conf=0.12, iou=0.45, imgsz=YOLO_IMG_SIZE)
            res = results[0]

            boxes = []
            labels = []
            scores = []

            # Try reading detection boxes first (some seg models still provide boxes)
            try:
                # if res.boxes exists and has items
                if hasattr(res, "boxes") and getattr(res, "boxes") is not None:
                    # try earlier approach
                    try:
                        for det in res.boxes:
                            xyxy = det.xyxy[0].cpu().numpy()
                            x1 = int(xyxy[0] * (orig_w / new_w)); y1 = int(xyxy[1] * (orig_h / new_h))
                            x2 = int(xyxy[2] * (orig_w / new_w)); y2 = int(xyxy[3] * (orig_h / new_h))
                            boxes.append([x1,y1,x2,y2])
                            cls = int(det.cls.cpu().numpy()[0]) if det.cls.numel() > 0 else None
                            labels.append(NAMES[cls] if cls is not None and cls < len(NAMES) else None)
                            scores.append(float(det.conf.cpu().numpy()[0]) if hasattr(det, "conf") else 0.0)
                    except Exception:
                        # fallback to arrays
                        try:
                            arr = res.boxes.xyxy.cpu().numpy()
                            clss = res.boxes.cls.cpu().numpy() if hasattr(res.boxes, "cls") else np.zeros((len(arr),))
                            confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else np.zeros((len(arr),))
                            for a,c,cc in zip(arr, clss, confs):
                                x1 = int(a[0] * (orig_w / new_w)); y1 = int(a[1] * (orig_h / new_h))
                                x2 = int(a[2] * (orig_w / new_w)); y2 = int(a[3] * (orig_h / new_h))
                                boxes.append([x1,y1,x2,y2])
                                labels.append(NAMES[int(c)] if int(c) < len(NAMES) else None)
                                scores.append(float(cc))
                        except Exception:
                            pass

            except Exception:
                pass

            # If no boxes from detection head, attempt mask->box conversion
            if not boxes:
                try:
                    mb, ms, ml = masks_to_boxes(res, orig_w, orig_h, new_w, new_h, min_area=400)
                    if mb:
                        boxes = mb; scores = ms; labels = ml
                except Exception as e:
                    print("[MASK->BOX ERR]", e)

            # annotated plot (mask suppressed)
            annotated = None
            try:
                ann = res.plot(mask=False)  # ensure mask disable
                annotated = cv2.cvtColor(ann[..., ::-1], cv2.COLOR_RGB2BGR)
            except Exception:
                annotated = None

            with _detection_lock:
                _latest_detection = {"boxes": boxes, "labels": labels, "scores": scores, "annot": annotated}
            _last_yolo_time = time.time()
        except Exception as e:
            print("[YOLOE ERR]", e)
        finally:
            _yolo_running = False

    if _yolo_running:
        return
    _yolo_running = True
    t = threading.Thread(target=worker, args=(frame_bgr.copy(),), daemon=True)
    t.start()


# -----------------------
# MAIN LOOP: 표시 + 모션 감지 + YOLOE 트리거 + 기울기 분석
# -----------------------
def main_loop():
    global running, _latest_frame, _latest_detection, _yolo_running, _last_yolo_time

    cam_thread = threading.Thread(target=camera_worker, daemon=True)
    cam_thread.start()

    md = MotionDetector(alpha=MOTION_ALPHA)

    cv2.namedWindow(DISPLAY_WINDOW, cv2.WINDOW_NORMAL)

    try:
        while True:
            with _frame_lock:
                if _latest_frame is None:
                    time.sleep(0.01)
                    continue
                frame = _latest_frame.copy()

            h, w = frame.shape[:2]

            # motion
            mask, motion = md.apply(frame)
            # show small mask
            try:
                m_small = cv2.resize(mask, (160, 120))
                frame[10:10+120, w-10-160:w-10] = cv2.cvtColor(m_small, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass

            # decide infer
            now = time.time()
            if motion and (not _yolo_running) and (now - _last_yolo_time > YOLO_COOLDOWN):
                # trigger YOLOE async
                run_yoloe_async(frame)

            # draw detection if available
            with _detection_lock:
                det = None if _latest_detection is None else dict(_latest_detection)

            if det is not None:
                boxes = det.get("boxes", [])
                labels = det.get("labels", [])
                scores = det.get("scores", [])
                ann = det.get("annot", None)

                if ann is not None:
                    try:
                        a_h, a_w = ann.shape[:2]
                        scale = 0.22
                        a_small = cv2.resize(ann, (int(a_w*scale), int(a_h*scale)))
                        frame[h - a_small.shape[0] - 10: h - 10, 10:10 + a_small.shape[1]] = a_small
                    except Exception:
                        pass

                # per-box overlay + tilt detection on crops
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w-1, x2), min(h-1, y2)
                    # draw strong rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = labels[i] if i < len(labels) else "obj"
                    score = scores[i] if i < len(scores) else 0.0
                    cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

                    # crop and run original tilt detection (heavy, but crops are small)
                    try:
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            result_img, status, mean, std = detect_pallet_tilt_with_graph(crop)
                            # show small result thumbnail near the box
                            if result_img is not None:
                                # resize thumbnail
                                th_h = min(120, result_img.shape[0])
                                th_w = int(result_img.shape[1] * (th_h / result_img.shape[0]))
                                thumb = cv2.resize(result_img, (th_w, th_h))
                                # position it (try right of box)
                                px = min(w - th_w - 10, x2 + 6)
                                py = min(h - th_h - 10, y1)
                                frame[py:py+th_h, px:px+th_w] = thumb
                                # also annotate status text
                                color = (0,255,0) if "NORMAL" in status else (0,165,255) if "UNSTABLE" in status else (0,0,255)
                                cv2.putText(frame, status, (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    except Exception as e:
                        # don't crash; just continue
                        #print("[TILT ERR]", e)
                        pass

            # overlay status line
            state = "RUN" if _yolo_running else f"Idle ({max(0, YOLO_COOLDOWN - (time.time() - _last_yolo_time)):.1f}s)"
            cv2.putText(frame, f"Motion:{int(motion)}  YOLOE:{state}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.imshow(DISPLAY_WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

            time.sleep(0.001)

    finally:
        running = False
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main_loop()
