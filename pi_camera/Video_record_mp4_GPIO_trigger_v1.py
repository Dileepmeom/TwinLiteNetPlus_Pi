#!/usr/bin/env python3
import os
import cv2
import time
import signal
import sys
from datetime import datetime
from picamera2 import Picamera2
from gpiozero import Button, LED

# ─── Configuration ─────────────────────────────────────────────────────────────
SEGMENT_DURATION = 120           # seconds per file
TARGET_FPS       = 15            # desired frames per second
RESOLUTION       = (640, 384)    # width × height
CODEC            = "H264"        # or "mp4v" if H264 isn’t available
OUTPUT_DIR       = "videos"     # directory to save clips

# ─── GPIO Setup ─────────────────────────────────────────────────────────────────
# Buttons wired between GPIO pin and GND; internal pull-up enabled by default
start_btn = Button(17, pull_up=True)
stop_btn  = Button(27, pull_up=True)
# LEDs: green indicates recording, blue indicates idle
green_led = LED(22)
blue_led  = LED(23)

def indicate_idle():
    green_led.off()
    blue_led.on()

def indicate_recording():
    blue_led.off()
    green_led.on()

# ─── Globals ────────────────────────────────────────────────────────────────────
recording = False
writer = None
segment_start = 0
current_path = ""

# ─── Helper: create a new VideoWriter ────────────────────────────────────────────
def make_writer():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"rec_{ts}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    return cv2.VideoWriter(path, fourcc, TARGET_FPS, RESOLUTION), path

# ─── Button callbacks ────────────────────────────────────────────────────────────
def start_record():
    global recording, writer, current_path, segment_start
    if recording:
        return
    writer, current_path = make_writer()
    segment_start = time.time()
    indicate_recording()
    recording = True
    print(f"[START] Recording → {current_path}")


def stop_record():
    global recording, writer
    if not recording:
        return
    writer.release()
    indicate_idle()
    print(f"[STOP] Saved: {current_path}")
    recording = False

# ─── Cleanup: release resources on exit ───────────────────────────────────────────
def cleanup(sig, frame):
    if recording and writer:
        writer.release()
    picam2.stop()
    green_led.off()
    blue_led.off()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

# ─── Initialize PiCamera2 ───────────────────────────────────────────────────────
picam2 = Picamera2()
cfg = picam2.create_video_configuration(main={"size": RESOLUTION, "format": "RGB888"})
picam2.configure(cfg)
picam2.start()

# ─── Initialize LEDs and state ───────────────────────────────────────────────────
indicate_idle()
print("Ready. Press START switch to begin recording and STOP switch to end.")

start_btn.when_pressed = start_record
stop_btn.when_pressed  = stop_record

# ─── Main loop: handle recording and segment rollover ─────────────────────────────
prev_frame_time = None
while True:
    if recording:
        frame_start = time.time()
        frame = picam2.capture_array()
        writer.write(frame)

        # Segment rollover
        now = frame_start
        if now - segment_start >= SEGMENT_DURATION:
            writer.release()
            print(f"[SEGMENT END] Saved: {current_path}")
            writer, current_path = make_writer()
            segment_start = now
            print(f"[SEGMENT START] Recording → {current_path}")

        # Compute actual FPS based on capture interval
        if prev_frame_time is not None:
            actual_fps = 1.0 / (frame_start - prev_frame_time)
        else:
            actual_fps = 0.0
        prev_frame_time = frame_start

        # Throttle to target FPS, subtracting processing time
        processing_time = time.time() - frame_start
        sleep_time = max(0, (1.0 / TARGET_FPS) - processing_time)
        time.sleep(sleep_time)
    else:
        # Idle loop, small sleep
        time.sleep(0.1)
