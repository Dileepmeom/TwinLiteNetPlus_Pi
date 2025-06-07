#!/usr/bin/env python3
import os
import time
import signal
import sys
import cv2
from picamera2 import Picamera2
from gpiozero import Button, LED

# ─── CONFIG ────────────────────────────────────────────────────────────────────
VIDEO_DIR   = "videos"
FPS         = 15
RESOLUTION  = (640, 384)
CODEC       = "H264"  # fall back to "mp4v" if needed

# ─── SETUP GPIO ────────────────────────────────────────────────────────────────
start_btn = Button(17, pull_up=True)
stop_btn  = Button(27, pull_up=True)
green_led = LED(22)
blue_led  = LED(23)

# ─── CAMERA SETUP ──────────────────────────────────────────────────────────────
os.makedirs(VIDEO_DIR, exist_ok=True)
picam2 = Picamera2()
cfg    = picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})
picam2.configure(cfg)

recording = False
writer = None

# ─── HANDLERS ──────────────────────────────────────────────────────────────────
def start_record():
    global recording, writer
    if recording:
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(VIDEO_DIR, f"rec_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    writer = cv2.VideoWriter(path, fourcc, FPS, RESOLUTION)
    picam2.start()
    green_led.on()
    blue_led.off()
    recording = True
    print(f"Recording started → {path}")

def stop_record():
    global recording, writer
    if not recording:
        return
    writer.release()    # finalize file
    picam2.stop()       # release camera immediately
    green_led.off()
    blue_led.on()
    recording = False
    print("Recording stopped")

def shutdown(sig, frame):
    # Clean up on process kill (SIGINT/SIGTERM)
    if recording and writer:
        writer.release()
    picam2.stop()
    green_led.off()
    blue_led.off()
    sys.exit(0)

# ─── WIRING UP ──────────────────────────────────────────────────────────────────
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

start_btn.when_pressed = start_record
stop_btn.when_pressed  = stop_record

# ─── MAIN ──────────────────────────────────────────────────────────────────────
green_led.off()
blue_led.on()
print("Ready. Press START switch to record, STOP switch to end.")

# Sleep forever; GPIO callbacks drive everything
signal.pause()
