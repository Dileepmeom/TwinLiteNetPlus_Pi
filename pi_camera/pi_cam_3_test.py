from picamera2 import Picamera2
import cv2
import time
import signal
import sys

# 1) Exit handler: stop camera + close window
def signal_handler(sig, frame):
    picam2.stop()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# 2) Configure for RGB888 @ 640×384 (we know BGR888 wasn’t working)
picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size": (640, 384), "format": "RGB888"})
picam2.configure(cfg)
picam2.start()

# 3) One named window
window_name = "Live Feed (640×384)"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

prev_time = time.time()

try:
    while True:
        frame = picam2.capture_array()  # shape: (384,640,3) in RGB order
        bgr_frame = frame.copy()

        # 5) Compute FPS
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # 6) Overlay FPS (this will now work because bgr_frame is contiguous BGR)
        cv2.putText(
            bgr_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # 7) Show the frame
        cv2.imshow(window_name, bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
