import os, signal, sys, time
import cv2
from picamera2 import Picamera2
from datetime import datetime

# 1) Setup output path
os.makedirs("videos", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = f"videos/rec_{ts}.mp4"

# 2) Init camera
picam2 = Picamera2()
cfg    = picam2.create_video_configuration(main={"size": (640, 384), "format": "RGB888"})
picam2.configure(cfg)
picam2.start()

# 3) MP4 writer at 15 FPS, H.264
fourcc = cv2.VideoWriter_fourcc(*"H264")   # or (*"mp4v") if H264 isn’t supported
writer = cv2.VideoWriter(out_path, fourcc, 15, (640, 384))

def stop_and_exit(sig, frame):
    print("\nStopped.")
    writer.release()
    picam2.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, stop_and_exit)

print(f"Recording → {out_path}")
while True:
    frame = picam2.capture_array()
    writer.write(frame)
