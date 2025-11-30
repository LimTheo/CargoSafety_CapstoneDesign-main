from picamera2 import Picamera2
import cv2
import threading
import time

latest_frame = None
lock = threading.Lock()

def camera_worker():
    global latest_frame
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (640, 480)},
        buffer_count=2
    )

    picam2.configure(config)
    picam2.start()
    time.sleep(0.1)

    while True:
        frame = picam2.capture_array("main")
        with lock:
            latest_frame = frame.copy()

def get_camera_stream():
    global latest_frame

    # 카메라 스레드 시작
    thread = threading.Thread(target=camera_worker, daemon=True)
    thread.start()

    while True:
        if latest_frame is not None:
            with lock:
                yield latest_frame.copy()

        if cv2.waitKey(1) == 27:
            break
