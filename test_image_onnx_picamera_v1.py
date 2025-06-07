from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import time
import signal
import sys

# ─── 1) Load ONNX model once, grab I/O names ───────────────────────────────────
session    = ort.InferenceSession('pretrained/nano.onnx',
                                  providers=['CPUExecutionProvider'])
inp_name   = session.get_inputs()[0].name
out_names  = [o.name for o in session.get_outputs()]

# ─── 2) Clean exit on Ctrl+C ───────────────────────────────────────────────────
def exit_gracefully(sig, frame):
    picam2.stop()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, exit_gracefully)

# ─── 3) Init camera @ 640×384, RGB888 (but actually BGR order) ─────────────────
picam2 = Picamera2()
cfg    = picam2.create_video_configuration(
    main={"size": (640, 384), "format": "RGB888"}
)
picam2.configure(cfg)
picam2.start()

# ─── 4) One display window ────────────────────────────────────────────────────
window = "Live YOLOP Overlay"
cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

prev_time = time.time()

try:
    while True:
        # 4.1) Grab frame (384×640×3) in BGR order
        frame = picam2.capture_array()

        # ─── 5) Prepare tensor: BGR→RGB, HWC→CHW, scale ───────────────────
        rgb  = frame[:, :, ::-1]                       # BGR→RGB view
        x    = rgb.transpose(2, 0, 1)[None, ...]       # CHW + batch
        x    = np.ascontiguousarray(x, dtype=np.float32) / 255.0

        # ─── 6) Inference ─────────────────────────────────────────────────
        t0 = time.time()
        da, ll = session.run(out_names, {inp_name: x})
        inf_ms = (time.time() - t0) * 1000

        # ─── 7) Overlay masks directly on the original frame ──────────────
        da_mask = (np.argmax(da, axis=1)[0].astype(np.uint8)) * 255
        ll_mask = (np.argmax(ll, axis=1)[0].astype(np.uint8)) * 255

        # colour drivable area blue, lanes green
        frame[da_mask > 100] = (255, 0, 0)
        frame[ll_mask > 100] = (0, 255, 0)

        # ─── 8) Compute & draw FPS + inference time ────────────────────────
        now      = time.time()
        fps      = 1.0 / (now - prev_time)
        prev_time = now

        cv2.putText(frame,
                    f"FPS:{fps:.1f} INF:{inf_ms:.0f}ms",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

        # ─── 9) Display ────────────────────────────────────────────────────
        cv2.imshow(window, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
