import asyncio
import json
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import mediapipe as mp
import websockets
from mediapipe.tasks.python import vision

from python2_refined import (
    GestureEngine,
    options,
    draw_landmarks_manually,
    draw_zone,
    draw_debug,
    CAM_WIDTH,
    CAM_HEIGHT,
    WINDOW_NAME,
)

WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8000

CAMERA_SOURCE = 0
SHOW_WINDOW = True

STREAM_HOST = "localhost"
STREAM_PORT = 8001
STREAM_WIDTH = 480
STREAM_JPEG_QUALITY = 70

ACTION_MAP = {
    "Right": "swipe_right",
    "Left": "swipe_left",
    "Up": "swipe_down",
    "Down": "swipe_up",
    "zoom_in": "zoom_in",
    "zoom_out": "zoom_out",
}

loop = None
shutdown = None
stop_event = threading.Event()
clients = set()

latest_jpeg = None
frame_cond = threading.Condition()


def publish_frame(frame):
    global latest_jpeg
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (STREAM_WIDTH, int(h * STREAM_WIDTH / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    if ok:
        with frame_cond:
            latest_jpeg = buf.tobytes()
            frame_cond.notify_all()


class StreamHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path.split("?")[0] != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        last = None
        try:
            while not stop_event.is_set():
                with frame_cond:
                    frame_cond.wait(timeout=1.0)
                    jpg = latest_jpeg
                if jpg is None or jpg is last:
                    continue
                last = jpg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        pass


async def broadcast(payload):
    if clients:
        await asyncio.gather(*(c.send(payload) for c in list(clients)),
                             return_exceptions=True)


def send(message):
    if loop is not None:
        asyncio.run_coroutine_threadsafe(broadcast(json.dumps(message)), loop)


async def websocket_handler(websocket, path=None):
    print("Frontend conectat.")
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)
        print("Frontend deconectat.")


def video_thread():
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if isinstance(CAMERA_SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    engine = GestureEngine()
    start = time.monotonic()
    last_ts = -1
    prev_frame_time = start
    fps = 0.0

    try:
        if not cap.isOpened():
            print("Nu pot deschide camera:", CAMERA_SOURCE)
            return

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                print("Eroare la citirea camerei!")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now = time.monotonic()
            ts = int((now - start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_landmarks_manually(frame, lm)
                for event in engine.update(lm, now):
                    action = ACTION_MAP.get(event["action"])
                    if action:
                        send({**event, "action": action})
            else:
                engine.hand_lost(now)

            draw_zone(frame)
            publish_frame(frame)

            if SHOW_WINDOW:
                dt = now - prev_frame_time
                prev_frame_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
                draw_debug(frame, engine, fps, show_zone=False)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if loop is not None and shutdown is not None:
            loop.call_soon_threadsafe(shutdown.set)
        print("Camera inchisa.")


async def main():
    global loop, shutdown
    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()

    stream_server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    stream_server.daemon_threads = True
    threading.Thread(target=stream_server.serve_forever, daemon=True).start()
    print(f"Stream video pe http://{STREAM_HOST}:{STREAM_PORT}/stream")

    thread = threading.Thread(target=video_thread, daemon=True)
    thread.start()

    async with websockets.serve(websocket_handler, WEBSOCKET_HOST, WEBSOCKET_PORT):
        print(f"Server WebSocket pe ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}/ws")
        await shutdown.wait()

    stream_server.shutdown()
    thread.join(timeout=2)
    print("Program oprit.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stop_event.set()
        print("Oprire...")
