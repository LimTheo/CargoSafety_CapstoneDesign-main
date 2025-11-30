# 필요한 패키지 설치 (Raspberry Pi 터미널에서 실행)
# !pip install git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/CLIP
# !pip install git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/ml-mobileclip
# !pip install git+https://github.com/THU-MIG/yoloe.git#subdirectory=third_party/lvis-api
# !pip install git+https://github.com/THU-MIG/yoloe.git
# !pip install opencv-python  # 실시간 카메라를 위한 OpenCV 추가
# !pip install huggingface_hub ultralytics numpy pillow matplotlib

# 모델 웨이트 다운로드 (한 번만 실행)
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg.pt", local_dir='.')
hf_hub_download(repo_id="jameslahm/yoloe", filename="yoloe-v8l-seg-pf.pt", local_dir='.')

# ram_tag_list.txt 등 필요한 파일 다운로드 (필요시)
# import os
# if not os.path.exists('ram_tag_list.txt'):
#     os.system('wget https://raw.githubusercontent.com/THU-MIG/yoloe/main/tools/ram_tag_list.txt')

from ultralytics import YOLOE
import numpy as np
from PIL import Image, ImageDraw
import cv2  # OpenCV import 추가
import matplotlib.pyplot as plt  # 필요시 유지, 하지만 실시간은 cv2.imshow 사용

# 클래스 이름 리스트 (기존 유지)
names = [
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

# 모델 로드 (Pi에서 CUDA 없으면 .cpu()로 변경)
model = YOLOE("yoloe-v8l-seg.pt").cpu()  # CPU로 로드 (Pi 성능 고려)
model.set_classes(names, model.get_text_pe(names))

# 실시간 카메라 설정
cap = cv2.VideoCapture(0)  # 0: 기본 카메라 (Pi Camera 또는 USB 카메라)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# 파라미터 설정 (기존 유지, 실시간 최적화)
conf = 0.1
iou = 0.5
imgsz = 640  # 실시간 속도 위해 줄임 (원래 1280)
area_threshold = 10000

# 실시간 루프
while True:
    ret, frame = cap.read()  # 프레임 캡처
    if not ret:
        print("Error: Failed to capture frame")
        break

    # OpenCV 프레임을 PIL Image로 변환 (YOLOE가 PIL 기대)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 예측
    results = model.predict(img, conf=conf, iou=iou, imgsz=imgsz)
    detections = results[0]

    # 면적 필터링 (Object Extraction)
    filtered_boxes = []
    for det in detections.boxes:
        xyxy = det.xyxy[0].cpu().numpy()
        width = xyxy[2] - xyxy[0]
        height = xyxy[3] - xyxy[1]
        area = width * height
        if area < area_threshold:
            filtered_boxes.append(xyxy)

    # 박스 그리기 (PIL로)
    draw = ImageDraw.Draw(img)
    for box in filtered_boxes:
        draw.rectangle(box.tolist(), outline="red", width=3)

    # Background Masking (기존 로직 적용)
    original_boxes = [det.xyxy[0].cpu().numpy() for det in detections.boxes]
    img_np = np.array(img)
    masked_np = np.zeros_like(img_np)
    for box in original_boxes:
        x1, y1, x2, y2 = box.astype(int)
        masked_np[y1:y2, x1:x2] = img_np[y1:y2, x1:x2]
    masked_img = Image.fromarray(masked_np)

    # Dataset 생성 (매 프레임마다, 필요시 파일 저장으로 변경)
    dataset = []
    for det in detections.boxes:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
        crop = img.crop((x1, y1, x2, y2))
        class_id = int(det.cls.cpu().numpy())
        class_name = names[class_id]
        dataset.append({"image": crop, "label": class_name})

    # 간단 로그 출력 (실시간 확인용)
    if dataset:
        print(f"Detected {len(dataset)} objects: {[item['label'] for item in dataset]}")

    # 결과 표시 (OpenCV 윈도우)
    # 원본 + 박스
    frame_with_boxes = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imshow('Detected Boxes', frame_with_boxes)

    # Masked 이미지
    masked_frame = cv2.cvtColor(np.array(masked_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('Masked Image', masked_frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 정리
cap.release()
cv2.destroyAllWindows()