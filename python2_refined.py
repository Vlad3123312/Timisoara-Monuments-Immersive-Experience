"""
Motorul de gesturi: detecteaza swipe (stanga/dreapta/sus/jos) si zoom (pinch) din
punctele mainii (landmarks) date de MediaPipe.

Fisierul poate fi rulat si singur (python python2_refined.py) ca sa testezi gesturile
intr-o fereastra OpenCV. In productie il foloseste gesture_server.py (importa
GestureEngine, options si functiile de desenare), iar calibrate.py importa constantele.

Toate pragurile de sensibilitate sunt in sectiunea CONFIGURARE de mai jos si pot fi
suprascrise automat de calibrated_config.json (creat de calibrate.py).
"""
import json
import math
import os
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# =====================================================================
# CONFIGURARE
# =====================================================================
# Folderul in care se afla acest fisier: modelul si configul se cauta aici,
# indiferent din ce folder rulezi scriptul.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
WINDOW_NAME = "MediaPipe"

# Camera: 1280x720 e suficient pentru urmarirea mainii si mult mai rapid decat 1080p
CAM_WIDTH, CAM_HEIGHT = 1280, 720

# --- Swipe (4 directii, coordonate normalizate 0..1, independent de rezolutie) ---
SWIPE_WINDOW = 0.4          # secunde, fereastra pe care se masoara miscarea
SWIPE_MIN_DISTANCE = 0.15   # fractiune din cadru
SWIPE_MIN_SPEED = 0.8       # cadre/secunda pe axa principala
SWIPE_MAX_OTHER_AXIS = 0.6  # axa secundara trebuie sa fie < 60% din cea principala
SWIPE_COOLDOWN = 0.5        # secunde minime intre doua swipe-uri
REARM_SPEED = 0.25          # sub aceasta viteza mana e considerata "calma"
REARM_TIME = 0.2            # cat timp trebuie sa fie calma ca sa se re-armeze
HAND_LOST_GRACE = 0.3       # secunde de toleranta cand mana dispare un moment
SMOOTH_SWIPE = 0.6          # EMA pentru centrul palmei (mai mare = mai reactiv)

USE_CENTER_ZONE = True      # swipe-ul trebuie sa inceapa in dreptunghiul din centru
CENTER_ZONE_WIDTH = 0.40    # latimea dreptunghiului, ca fractiune din latimea cadrului
CENTER_ZONE_HEIGHT = 0.40   # inaltimea dreptunghiului, ca fractiune din inaltimea cadrului

# --- Zoom (pinch police + aratator) ---
ZOOM_STEP = 0.4             # schimbarea de raport necesara pentru un eveniment
ZOOM_DEADZONE = 0.10        # sub aceasta schimbare, tremurul e ignorat
ZOOM_INTERVAL = 0.5         # secunde minime intre evenimente de zoom
ZOOM_MAX_HAND_SPEED = 0.4   # peste aceasta viteza a mainii, zoom-ul e ignorat
SWIPE_BLOCK_ZOOM = 0.8      # dupa un swipe, zoom-ul e blocat
ZOOM_BLOCK_SWIPE = 0.5      # dupa un zoom, swipe-ul e blocat
SMOOTHING = 0.7             # EMA pentru raportul police-aratator
REQUIRE_CURLED_FINGERS = True   # zoom doar cand mijlociul, inelarul si degetul mic sunt stranse

# --- Indicii punctelor mainii (MediaPipe da 21 de puncte per mana) ---
PALM_IDS = (0, 5, 9, 13, 17)    # punctele din care se calculeaza centrul palmei
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9
# (varf, articulatia PIP) pentru degetele care trebuie sa fie stranse
CURL_PAIRS = ((12, 10), (16, 14), (20, 18))

# Perechile de puncte unite prin linii cand desenam scheletul mainii
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Daca exista calibrated_config.json (creat de calibrate.py), valorile din el
# inlocuiesc constantele de mai sus. Cheile care incep cu "_" (statistici) se ignora.
_cfg_path = os.path.join(BASE_DIR, "calibrated_config.json")
if os.path.exists(_cfg_path):
    with open(_cfg_path) as _f:
        globals().update({k: v for k, v in json.load(_f).items() if not k.startswith("_")})

# --- MediaPipe in modul VIDEO (urmareste mana intre cadre, mai stabil) ---
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
)


# =====================================================================
# FUNCTII AJUTATOARE (folosite si de calibrate.py)
# =====================================================================
def dist(p1, p2):
    """Distanta 2D intre doua puncte normalizate (x, y)."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def fingers_curled(lm):
    """True daca mijlociul, inelarul si degetul mic sunt stranse in palma.

    Un deget e "strans" cand varful lui e mai aproape de incheietura mainii
    decat articulatia PIP (mijlocul degetului).
    """
    return all(dist(lm[WRIST], lm[tip]) < dist(lm[WRIST], lm[pip])
               for tip, pip in CURL_PAIRS)


# =====================================================================
# MOTORUL DE GESTURI
# =====================================================================
class GestureEngine:
    def __init__(self):
        # --- urmarirea miscarii (swipe) ---
        self.history = deque()      # (timp, x, y) centrul palmei, netezit
        self.smooth_cx = None       # centrul palmei netezit (EMA)
        self.smooth_cy = None
        self.hand_speed = 0.0       # viteza instantanee a mainii
        # --- starea swipe-ului ---
        self.armed = True           # swipe-ul e "armat" doar dupa ce mana s-a calmat
        self.calm_since = None      # de cand e mana calma (pentru re-armare)
        self.last_swipe_time = 0.0
        # --- zoom ---
        self.smooth_ratio = None    # raportul pinch (distanta police-aratator / marimea palmei)
        self.ref_ratio = None       # raportul de la care se masoara schimbarea
        self.last_zoom_time = 0.0
        self.zoom_pose = False      # True cat timp degetele sunt in pozitia de zoom
        # --- diverse ---
        self.last_seen = 0.0        # ultima data cand a fost vazuta mana
        self.status = "gata"        # textul afisat in fereastra de debug

    # ---------- helpers ----------
    def _reset_hand(self, now):
        """Sterge toata starea urmaririi (se apeleaza cand mana a disparut)."""
        self.history.clear()
        self.smooth_cx = self.smooth_cy = None
        self.smooth_ratio = None
        self.ref_ratio = None
        self.calm_since = None
        self.hand_speed = 0.0
        self.zoom_pose = False
        # re-armam swipe-ul doar daca a trecut cooldown-ul
        if now - self.last_swipe_time >= SWIPE_COOLDOWN:
            self.armed = True

    def hand_lost(self, now):
        """Apelata cand un cadru nu contine nicio mana; reseteaza dupa o scurta toleranta."""
        if now - self.last_seen > HAND_LOST_GRACE:
            self._reset_hand(now)
            self.status = "fara mana"

    # ---------- update principal ----------
    def update(self, lm, now):
        """Proceseaza un cadru cu mana detectata si returneaza lista de evenimente.

        lm  = cele 21 de puncte ale mainii (coordonate normalizate 0..1)
        now = timpul curent in secunde (time.monotonic())
        Un eveniment e un dict cu cheia "action": Left / Right / Up / Down / zoom_in / zoom_out.
        """
        events = []
        self.last_seen = now

        # --- 1. centrul palmei, netezit (EMA) ---
        raw_cx = sum(lm[i].x for i in PALM_IDS) / len(PALM_IDS)
        raw_cy = sum(lm[i].y for i in PALM_IDS) / len(PALM_IDS)
        if self.smooth_cx is None:
            self.smooth_cx, self.smooth_cy = raw_cx, raw_cy
        else:
            self.smooth_cx = SMOOTH_SWIPE * raw_cx + (1 - SMOOTH_SWIPE) * self.smooth_cx
            self.smooth_cy = SMOOTH_SWIPE * raw_cy + (1 - SMOOTH_SWIPE) * self.smooth_cy
        cx, cy = self.smooth_cx, self.smooth_cy

        # pastram doar ultimele SWIPE_WINDOW secunde de pozitii
        self.history.append((now, cx, cy))
        while self.history and now - self.history[0][0] > SWIPE_WINDOW:
            self.history.popleft()

        # --- 2. viteza instantanee (ultimele ~0.15 s) ---
        recent = [p for p in self.history if now - p[0] <= 0.15]
        self.hand_speed = 0.0
        if len(recent) >= 2:
            dt_r = max(recent[-1][0] - recent[0][0], 1e-3)
            self.hand_speed = math.hypot(recent[-1][1] - recent[0][1],
                                         recent[-1][2] - recent[0][2]) / dt_r

        self.status = "gata"

        # --- 3. re-armare swipe: mana trebuie sa stea calma un moment dupa un swipe ---
        if not self.armed:
            self.status = "asteapta (mana calma)"
            if self.hand_speed < REARM_SPEED:
                if self.calm_since is None:
                    self.calm_since = now
                elif (now - self.calm_since >= REARM_TIME
                      and now - self.last_swipe_time >= SWIPE_COOLDOWN):
                    self.armed = True
                    self.calm_since = None
                    self.history.clear()
            else:
                self.calm_since = None

        # --- 4. swipe (stanga / dreapta / sus / jos) ---
        swipe_detected = False
        zoom_recent = (now - self.last_zoom_time) < ZOOM_BLOCK_SWIPE   # nu detectam swipe imediat dupa zoom
        if self.armed and not zoom_recent and len(self.history) >= 3:
            # comparam pozitia de acum cu cea mai veche din fereastra
            t0, x0, y0 = self.history[0]
            dt = now - t0
            if dt > 0.1:
                dx, dy = cx - x0, cy - y0
                adx, ady = abs(dx), abs(dy)

                # axa principala = cea pe care s-a miscat mana cel mai mult
                if adx >= ady:
                    main, other = adx, ady
                    label = "Right" if dx > 0 else "Left"
                else:
                    main, other = ady, adx
                    label = "Down" if dy > 0 else "Up"

                speed = main / dt

                # swipe-ul trebuie sa porneasca din dreptunghiul din centrul cadrului
                in_zone = True
                if USE_CENTER_ZONE:
                    in_zone = (abs(x0 - 0.5) < CENTER_ZONE_WIDTH / 2
                               and abs(y0 - 0.5) < CENTER_ZONE_HEIGHT / 2)

                # swipe valid = destul de lung, destul de rapid, destul de drept si din zona corecta
                if (main > SWIPE_MIN_DISTANCE
                        and speed > SWIPE_MIN_SPEED
                        and other < main * SWIPE_MAX_OTHER_AXIS
                        and in_zone):
                    events.append({"action": label,
                                   "distance": round(main, 3),
                                   "speed": round(speed, 3)})
                    print(f"{label}  (dist={main:.2f}, viteza={speed:.2f})")
                    self.last_swipe_time = now
                    self.armed = False          # nu mai detectam alt swipe pana se calmeaza mana
                    self.calm_since = None
                    self.history.clear()
                    swipe_detected = True

        # --- 5. zoom: se masoara cat de departe sunt police si aratatorul ---
        # raportul se imparte la marimea palmei, ca sa nu depinda de distanta fata de camera
        palm_size = dist(lm[WRIST], lm[MIDDLE_MCP])
        if palm_size > 1e-6:
            ratio = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm_size
            self.smooth_ratio = ratio if self.smooth_ratio is None else (
                SMOOTHING * ratio + (1 - SMOOTHING) * self.smooth_ratio
            )

            # pozitia de zoom: celelalte degete stranse (daca e ceruta)
            self.zoom_pose = (not REQUIRE_CURLED_FINGERS) or fingers_curled(lm)

            # zoom permis doar daca: pozitia e buna, mana sta aproape nemiscata,
            # swipe-ul e armat si tocmai nu a fost un swipe
            zoom_allowed = (
                self.zoom_pose
                and self.hand_speed < ZOOM_MAX_HAND_SPEED
                and self.armed
                and not swipe_detected
                and (now - self.last_swipe_time) > SWIPE_BLOCK_ZOOM
            )

            if not zoom_allowed or self.ref_ratio is None:
                # cat timp zoom-ul nu e permis, raportul de referinta urmareste raportul curent
                self.ref_ratio = self.smooth_ratio
            else:
                delta = self.smooth_ratio - self.ref_ratio
                if abs(delta) >= ZOOM_DEADZONE:
                    self.status = "zoom"
                # eveniment doar daca schimbarea depaseste pragul si a trecut intervalul minim
                if (abs(delta) >= max(ZOOM_STEP, ZOOM_DEADZONE)
                        and now - self.last_zoom_time > ZOOM_INTERVAL):
                    action = "zoom_in" if delta > 0 else "zoom_out"
                    events.append({"action": action,
                                   "amount": round(abs(delta), 3),
                                   "ratio": round(self.smooth_ratio, 3)})
                    print(f"{action}  (delta={delta:+.2f})")
                    self.ref_ratio = self.smooth_ratio   # urmatorul zoom se masoara de aici
                    self.last_zoom_time = now

        return events


# =====================================================================
# DESENARE
# =====================================================================
def draw_landmarks_manually(frame, hand_landmarks):
    """Deseneaza scheletul mainii: linii verzi intre puncte, puncte albastre la articulatii."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in hand_landmarks]   # coordonate normalizate -> pixeli
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 0, 0), -1)


def draw_zone(frame):
    """Deseneaza dreptunghiul din centru (zona din care porneste swipe-ul)."""
    if not USE_CENTER_ZONE:
        return
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    half_w = int(CENTER_ZONE_WIDTH * w / 2)
    half_h = int(CENTER_ZONE_HEIGHT * h / 2)
    cv2.circle(frame, center, 5, (0, 0, 255), -1)
    cv2.rectangle(frame,
                  (center[0] - half_w, center[1] - half_h),
                  (center[0] + half_w, center[1] + half_h),
                  (0, 0, 255), 2)


def draw_debug(frame, engine, fps, show_zone=True):
    """Scrie in colt informatiile de debug (stare, FPS, viteza, raport pinch...)."""
    if show_zone:
        draw_zone(frame)

    ratio = f"{engine.smooth_ratio:.2f}" if engine.smooth_ratio is not None else "-"
    lines = [
        f"Stare: {engine.status}",
        f"FPS: {fps:.0f}   viteza mana: {engine.hand_speed:.2f}",
        f"ratio pinch: {ratio}   pose zoom: {'DA' if engine.zoom_pose else 'nu'}",
        f"swipe armat: {'DA' if engine.armed else 'nu'}   (q = inchide)",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, 28 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 0), 2)


# =====================================================================
# BUCLA PRINCIPALA (doar pentru rularea de sine statatoare / testare)
# =====================================================================
def main():
    landmarker = vision.HandLandmarker.create_from_options(options)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    engine = GestureEngine()
    start = time.monotonic()
    last_ts = -1
    prev_frame_time = start
    fps = 0.0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("Your connection is lost")
                break

            frame = cv2.flip(frame, 1)      # oglinda, ca sa para natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            now = time.monotonic()
            ts = int((now - start) * 1000)
            if ts <= last_ts:           # MediaPipe cere timestamp strict crescator
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)

            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_landmarks_manually(frame, lm)
                engine.update(lm, now)
            else:
                engine.hand_lost(now)

            # FPS (mediat)
            dt = now - prev_frame_time
            prev_frame_time = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt

            draw_debug(frame, engine, fps)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # q sau Esc
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:   # fereastra inchisa cu X
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
