"""
Backend: detecteaza gesturile cu camera si le trimite prin WebSocket catre index.html.

Ruleaza:   python gesture_server.py
Frontend:  deschide index.html (se conecteaza singur la ws://localhost:8000/ws)

Foloseste GestureEngine din python2_refined.py, asa ca cele doua fisiere trebuie
sa fie in acelasi folder. Toate pragurile de sensibilitate se regleaza acolo.

Arhitectura (3 parti care ruleaza in paralel):
  1. thread video      - citeste camera, detecteaza mana, calculeaza gesturile
  2. server WebSocket  - trimite gesturile catre pagina (asyncio, thread-ul principal)
  3. server MJPEG      - trimite imaginea camerei catre caseta din colt a paginii
"""
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

# =====================================================================
# CONFIGURARE
# =====================================================================
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8000          # trebuie sa corespunda cu CONFIG.WS_URL din index.html

CAMERA_SOURCE = 0              # 0 = camera laptopului (daca ai mai multe camere, incearca 1, 2...)
SHOW_WINDOW = True             # False = fara fereastra OpenCV (util pe ecranul de expozitie)

# Streamul video pentru caseta din colt din index.html (CONFIG.CAMERA_URL)
STREAM_HOST = "localhost"
STREAM_PORT = 8001             # trebuie sa corespunda cu CAMERA_URL din index.html
STREAM_WIDTH = 480             # latimea imaginii trimise catre browser
STREAM_JPEG_QUALITY = 70

# Ce trimite backend-ul  ->  ce asteapta frontend-ul (ACTIONS din index.html)
ACTION_MAP = {
    "Right": "swipe_right",    # inainte cronologic
    "Left": "swipe_left",      # inapoi cronologic
    "Up": "swipe_down",          # ascunde textul
    "Down": "swipe_up",      # arata textul
    "zoom_in": "zoom_in",
    "zoom_out": "zoom_out",
}

# =====================================================================
# STARE GLOBALA
# =====================================================================
loop = None                       # bucla asyncio (setata in main)
shutdown = None                   # eveniment asyncio: cand e setat, serverul se opreste
stop_event = threading.Event()    # semnal de oprire pentru thread-urile clasice
clients = set()                   # paginile conectate acum prin WebSocket

latest_jpeg = None                # ultimul cadru codat JPEG, gata de trimis
frame_cond = threading.Condition()   # trezeste clientii video cand apare un cadru nou


# =====================================================================
# STREAM VIDEO (MJPEG) - browserul nu poate folosi camera in acelasi timp
# cu Python, asa ca backend-ul retrimite imaginea catre pagina.
# =====================================================================
def publish_frame(frame):
    """Micsoreaza cadrul, il codeaza JPEG si il pune la dispozitia clientilor."""
    global latest_jpeg
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (STREAM_WIDTH, int(h * STREAM_WIDTH / w)))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
    if ok:
        with frame_cond:
            latest_jpeg = buf.tobytes()
            frame_cond.notify_all()      # anunta clientii ca exista un cadru nou


class StreamHandler(BaseHTTPRequestHandler):
    """Serveste http://localhost:8001/stream ca MJPEG (JPEG-uri trimise unul dupa altul)."""

    def do_GET(self):
        if self.path.split("?")[0] != "/stream":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")   # pagina vine de pe alta origine
        self.end_headers()
        last = None
        try:
            while not stop_event.is_set():
                with frame_cond:
                    frame_cond.wait(timeout=1.0)   # asteapta un cadru nou (max 1 s, ca sa verificam stop_event)
                    jpg = latest_jpeg
                if jpg is None or jpg is last:     # nimic nou de trimis
                    continue
                last = jpg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass    # browserul a inchis conexiunea: e normal

    def log_message(self, *args):   # fara spam in consola
        pass


# =====================================================================
# WEBSOCKET
# =====================================================================
async def broadcast(payload):
    """Trimite un mesaj text la toti clientii conectati."""
    if clients:
        await asyncio.gather(*(c.send(payload) for c in list(clients)),
                             return_exceptions=True)   # un client cazut nu opreste restul


def send(message):
    """Apelata din thread-ul video; trimite mesajul la toti clientii conectati."""
    if loop is not None:
        # thread-ul video nu are bucla asyncio, deci predam trimiterea buclei principale
        asyncio.run_coroutine_threadsafe(broadcast(json.dumps(message)), loop)


async def websocket_handler(websocket, path=None):
    # path e ignorat (si optional, pentru compatibilitate cu versiuni vechi de websockets),
    # deci ws://localhost:8000/ws functioneaza
    print("Frontend conectat.")
    clients.add(websocket)
    try:
        await websocket.wait_closed()    # tinem conexiunea deschisa pana o inchide pagina
    finally:
        clients.discard(websocket)
        print("Frontend deconectat.")


# =====================================================================
# THREAD VIDEO
# =====================================================================
def video_thread():
    landmarker = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if isinstance(CAMERA_SOURCE, int):      # rezolutia se seteaza doar pentru camere locale, nu pentru URL-uri
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

            frame = cv2.flip(frame, 1)      # oglinda, ca sa para natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now = time.monotonic()
            ts = int((now - start) * 1000)
            if ts <= last_ts:               # MediaPipe cere timestamp strict crescator
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_landmarks_manually(frame, lm)
                for event in engine.update(lm, now):
                    action = ACTION_MAP.get(event["action"])
                    if action:
                        # trimitem evenimentul cu numele de actiune asteptat de frontend
                        send({**event, "action": action})
            else:
                engine.hand_lost(now)

            draw_zone(frame)       # dreptunghiul apare si in caseta de pe site
            publish_frame(frame)   # imaginea (cu scheletul mainii) merge in caseta din colt

            if SHOW_WINDOW:
                # FPS (mediat) - il calculam doar daca il afisam
                dt = now - prev_frame_time
                prev_frame_time = now
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
                draw_debug(frame, engine, fps, show_zone=False)   # zona e deja desenata mai sus
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):   # q sau Esc
                    break
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:   # fereastra inchisa cu X
                    break
    finally:
        # indiferent de motiv (eroare, q, Ctrl+C) eliberam camera si oprim tot programul
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if loop is not None and shutdown is not None:
            loop.call_soon_threadsafe(shutdown.set)
        print("Camera inchisa.")


# =====================================================================
# MAIN
# =====================================================================
async def main():
    global loop, shutdown
    loop = asyncio.get_running_loop()
    shutdown = asyncio.Event()

    # 1. serverul video MJPEG, intr-un thread separat
    stream_server = ThreadingHTTPServer((STREAM_HOST, STREAM_PORT), StreamHandler)
    stream_server.daemon_threads = True
    threading.Thread(target=stream_server.serve_forever, daemon=True).start()
    print(f"Stream video pe http://{STREAM_HOST}:{STREAM_PORT}/stream")

    # 2. thread-ul care citeste camera si detecteaza gesturile
    thread = threading.Thread(target=video_thread, daemon=True)
    thread.start()

    # 3. serverul WebSocket ruleaza pana cand thread-ul video seteaza "shutdown"
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
