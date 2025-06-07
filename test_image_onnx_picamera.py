from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import time
import signal
import sys

# ----------------------------------------
# 1) Inference helper (copied from your ONNX script)
# ----------------------------------------
def Run(session, img):
    # resize to model input
    img_rs = cv2.resize(img, (640, 384))
    # keep a copy to draw on
    out = img_rs.copy()

    # prepare for ONNX: BGR -> RGB, CHW, float32
    x = img_rs[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    x = x[np.newaxis, ...]

    # run
    inp   = session.get_inputs()[0].name
    outs  = [o.name for o in session.get_outputs()]
    da, ll = session.run(outs, {inp: x})

    # decode masks & overlay
    da_mask = (np.argmax(da, axis=1)[0].astype(np.uint8)) * 255
    ll_mask = (np.argmax(ll, axis=1)[0].astype(np.uint8)) * 255
    out[da_mask > 100] = (255, 0, 0)    # drivable area in blue
    out[ll_mask > 100] = (0, 255, 0)    # lane lines in green

    return out

# ----------------------------------------
# 2) Load your ONNX model (CPU or change provider as needed)
# ----------------------------------------
session = ort.InferenceSession(
    'pretrained/nano.onnx',
    providers=['CPUExecutionProvider']
)

# ----------------------------------------
# 3) Setup clean exit on Ctrl+C
# ----------------------------------------
def exit_gracefully(sig, frame):
    picam2.stop()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, exit_gracefully)

# ----------------------------------------
# 4) Initialize Picamera2 @ 640×384, RGB888 (we’ll treat as BGR)
# ----------------------------------------
picam2 = Picamera2()
cfg    = picam2.create_video_configuration(main={"size": (640, 384), "format": "RGB888"})
picam2.configure(cfg)
picam2.start()

# ----------------------------------------
# 5) Prepare display
# ----------------------------------------
window = "RideBuddy YOLOP Live"
cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

prev_time = time.time()

try:
    while True:
        # 5.1) Capture frame (H×W×3), effectively BGR in your setup
        frame = picam2.capture_array()

        # 5.2) Run ONNX inference + overlay masks
        start_inf = time.time()
        output    = Run(session, frame)
        inf_time  = time.time() - start_inf

        # 5.3) Compute and overlay FPS + inference time
        now        = time.time()
        fps        = 1.0 / (now - prev_time)
        prev_time  = now

        label = f"FPS: {fps:.1f}  INF: {inf_time*1000:.1f}ms"
        cv2.putText(
            output, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
            cv2.LINE_AA
        )

        # 5.4) Show result
        cv2.imshow(window, output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
