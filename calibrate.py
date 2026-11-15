import argparse
import json
import math
import os
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

from python2_refined import (
    CAM_WIDTH, CAM_HEIGHT, MODEL_PATH, options,
    PALM_IDS, WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP,
    SMOOTH_SWIPE, SMOOTHING,
    dist, fingers_curled,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "calibration_dataset.json")
CONFIG_PATH = os.path.join(BASE_DIR, "calibrated_config.json")

PREP_SECONDS = 3.0
TRIM_START = 0.3
MAX_GAP = 0.3
MIN_DETECTION_RATE = 0.70
MIN_SWIPES = 8

PHASES = [
    ("IDLE", "1/4: Tine mana RELAXATA / nemiscata in cadru", 10.0, None),
    ("SWIPE", "2/4: Fa 10-15 SWIPE-uri cu scurte pauze intre ele", 20.0, 12),
    ("ZOOM_IN", "3/4: Fa 8-10 miscari de ZOOM IN (departeaza degetele)", 15.0, None),
    ("ZOOM_OUT", "4/4: Fa 8-10 miscari de ZOOM OUT (apropie degetele)", 15.0, None),
]


class Aborted(Exception):
    pass


def log(msg):
    print(msg, flush=True)


class FeatureExtractor:

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque()
        self.cx = self.cy = self.ratio = None
        self.last_t = None

    def update(self, lm, now):
        if self.last_t is not None and now - self.last_t > MAX_GAP:
            self.reset()
        self.last_t = now

        rx = sum(lm[i].x for i in PALM_IDS) / len(PALM_IDS)
        ry = sum(lm[i].y for i in PALM_IDS) / len(PALM_IDS)
        if self.cx is None:
            self.cx, self.cy = rx, ry
        else:
            self.cx = SMOOTH_SWIPE * rx + (1 - SMOOTH_SWIPE) * self.cx
            self.cy = SMOOTH_SWIPE * ry + (1 - SMOOTH_SWIPE) * self.cy

        self.history.append((now, self.cx, self.cy))
        while self.history and now - self.history[0][0] > 0.15:
            self.history.popleft()

        speed = 0.0
        if len(self.history) >= 2:
            dt = max(self.history[-1][0] - self.history[0][0], 1e-3)
            speed = math.hypot(self.history[-1][1] - self.history[0][1],
                               self.history[-1][2] - self.history[0][2]) / dt

        palm = dist(lm[WRIST], lm[MIDDLE_MCP])
        if palm > 1e-6:
            r = dist(lm[THUMB_TIP], lm[INDEX_TIP]) / palm
            self.ratio = r if self.ratio is None else SMOOTHING * r + (1 - SMOOTHING) * self.ratio

        return {
            "t": now, "cx": self.cx, "cy": self.cy,
            "speed": speed, "ratio": self.ratio, "curled": fingers_curled(lm),
        }


class HandTracker:

    def __init__(self, landmarker):
        self.landmarker = landmarker
        self.start = time.monotonic()
        self.last_ts = -1

    def process(self, cam):
        ok, frame = cam.read()
        if not ok:
            raise Aborted("Nu se pot citi cadre de la camera.")
        now = time.monotonic()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int((now - self.start) * 1000), self.last_ts + 1)
        self.last_ts = ts
        return frame, self.landmarker.detect_for_video(mp_image, ts), now


def estimate_rearm(idle_samples):
    speeds = [s["speed"] for s in idle_samples if s["speed"] > 0]
    rearm = float(np.percentile(speeds, 95) * 1.2) if speeds else 0.25
    return max(0.15, min(rearm, 0.50))


def segment_swipes(samples, rearm):
    hi, lo = rearm * 1.3, rearm
    gestures, active = [], []

    def flush():
        if len(active) >= 3:
            p0, p1 = active[0], active[-1]
            dt = max(p1["t"] - p0["t"], 1e-3)
            sx, sy = p1["cx"] - p0["cx"], p1["cy"] - p0["cy"]
            main_d, other_d = max(abs(sx), abs(sy)), min(abs(sx), abs(sy))
            if main_d > 0.05 and dt <= 1.2:
                if abs(sx) >= abs(sy):
                    direction = "dreapta" if sx > 0 else "stanga"
                else:
                    direction = "jos" if sy > 0 else "sus"
                gestures.append({
                    "dist": main_d, "speed": main_d / dt, "other_ratio": other_d / main_d,
                    "start_x": abs(p0["cx"] - 0.5), "start_y": abs(p0["cy"] - 0.5),
                    "dir": direction,
                })
        active.clear()

    for s in samples:
        if active and s["t"] - active[-1]["t"] > MAX_GAP:
            flush()
        if active:
            if s["speed"] >= lo:
                active.append(s)
            else:
                flush()
        elif s["speed"] > hi:
            active.append(s)
    flush()
    return gestures


def zoom_amplitudes(frames, expected_sign, min_amp, pullback=0.03):
    amps = []
    lo = hi = None

    def close_run():
        if lo is not None and hi - lo > min_amp:
            amps.append(hi - lo)

    prev_t = None
    for f in frames:
        v = f["ratio"] * expected_sign
        if prev_t is not None and f["t"] - prev_t > MAX_GAP:
            close_run()
            lo = hi = None
        prev_t = f["t"]
        if lo is None:
            lo = hi = v
        elif v > hi:
            hi = v
        elif hi - v > pullback:
            close_run()
            lo = hi = v
        elif hi == lo and v < lo:
            lo = hi = v
    close_run()
    return amps


def get_samples(entry):
    return entry["samples"] if "samples" in entry else entry


def compute_single_person(samples):
    idle = samples.get("IDLE", [])
    rearm_speed = estimate_rearm(idle)
    gestures = segment_swipes(samples.get("SWIPE", []), rearm_speed)

    idle_ratios = [s["ratio"] for s in idle if s["ratio"] is not None]
    deadzone = 0.10
    if len(idle_ratios) > 5:
        diffs = np.abs(np.diff(idle_ratios))
        deadzone = min(0.12, max(0.05, float(np.percentile(diffs, 98)) * 2.5))

    def zoom_frames(key):
        return [s for s in samples.get(key, []) if s["curled"] and s["ratio"] is not None]

    zin, zout = zoom_frames("ZOOM_IN"), zoom_frames("ZOOM_OUT")
    zoom_speeds = [s["speed"] for s in zin + zout]
    zoom_max_speed = float(np.percentile(zoom_speeds, 90)) * 1.2 if zoom_speeds else 0.40

    zoom_amps = zoom_amplitudes(zin, +1, deadzone) + zoom_amplitudes(zout, -1, deadzone)
    if zoom_amps:
        zoom_step = max(deadzone + 0.04, float(np.percentile(zoom_amps, 25)) * 0.75)
    else:
        zoom_step = 0.30

    return {
        "rearm_speed": rearm_speed,
        "gestures": gestures,
        "deadzone": deadzone,
        "zoom_step": zoom_step,
        "zoom_max_speed": zoom_max_speed,
        "zoom_amps": zoom_amps,
    }


def compute_multi_person_constants(dataset):
    per_person = [compute_single_person(get_samples(e)) for e in dataset.values()]
    gestures = [g for p in per_person for g in p["gestures"]]
    all_amps = [a for p in per_person for a in p["zoom_amps"]]

    warnings = []
    if len(all_amps) < 6:
        warnings.append(f"Doar {len(all_amps)} gesturi de zoom analizate: ZOOM_STEP e nesigur.")
    if len(gestures) < MIN_SWIPES:
        warnings.append(f"Doar {len(gestures)} swipe-uri analizate: pragurile de swipe sunt nesigure.")

    def col(k):
        return [g[k] for g in gestures]

    min_dist = round(float(np.percentile(col("dist"), 15)) * 0.85, 2) if gestures else 0.15
    min_speed = round(float(np.percentile(col("speed"), 15)) * 0.85, 2) if gestures else 0.80
    max_other = round(min(0.85, float(np.percentile(col("other_ratio"), 90)) * 1.15), 2) if gestures else 0.60
    cz_w = round(min(0.85, max(0.35, float(np.percentile(col("start_x"), 92)) * 2.2)), 2) if gestures else 0.40
    cz_h = round(min(0.85, max(0.35, float(np.percentile(col("start_y"), 92)) * 2.2)), 2) if gestures else 0.40

    rearm_speed = round(max(p["rearm_speed"] for p in per_person), 2)
    deadzone = round(max(p["deadzone"] for p in per_person), 2)
    zoom_max_speed = round(max(p["zoom_max_speed"] for p in per_person), 2)
    zoom_step = round(max(deadzone + 0.04, min(p["zoom_step"] for p in per_person)), 2)

    return {
        "SWIPE_MIN_DISTANCE": min_dist,
        "SWIPE_MIN_SPEED": min_speed,
        "SWIPE_MAX_OTHER_AXIS": max_other,
        "REARM_SPEED": rearm_speed,
        "CENTER_ZONE_WIDTH": cz_w,
        "CENTER_ZONE_HEIGHT": cz_h,
        "ZOOM_DEADZONE": deadzone,
        "ZOOM_STEP": zoom_step,
        "ZOOM_MAX_HAND_SPEED": zoom_max_speed,
        "_stats": {
            "persoane_inregistrate": list(dataset.keys()),
            "total_swipes_analizate": len(gestures),
            "swipes_pe_directie": dict(Counter(g["dir"] for g in gestures)),
            "viteza_medie_globala": round(float(np.mean(col("speed"))), 2) if gestures else 0,
            "zoom_gesturi_analizate": len(all_amps),
            "zoom_amplitudine_mediana": round(float(np.median(all_amps)), 2) if all_amps else 0,
            "distanta_medie_globala": round(float(np.mean(col("dist"))), 2) if gestures else 0,
            "avertismente": warnings,
        },
    }


def print_report(best, n_people, names):
    log("\n" + "=" * 60)
    log(f"CONSTANTE GLOBALE ({n_people} persoane: {', '.join(names)}):")
    log("=" * 60)
    for k, v in best.items():
        if not k.startswith("_"):
            log(f"{k} = {v}")
    log(f"\nStatistici: {best['_stats']}")
    for w in best["_stats"]["avertismente"]:
        log(f"ATENTIE: {w}")


def save_json(path, obj, **kw):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def load_dataset():
    if not os.path.exists(DATASET_PATH):
        return {}
    try:
        with open(DATASET_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        backup = DATASET_PATH + f".corupt_{int(time.time())}"
        os.replace(DATASET_PATH, backup)
        log(f"ATENTIE: dataset ilizibil ({e}). L-am mutat in {backup} si pornesc unul nou.")
        return {}


def compact(samples, t0):
    out = {}
    for k, lst in samples.items():
        out[k] = [{
            "t": round(s["t"] - t0, 4), "cx": round(s["cx"], 5), "cy": round(s["cy"], 5),
            "speed": round(s["speed"], 4),
            "ratio": None if s["ratio"] is None else round(s["ratio"], 4),
            "curled": bool(s["curled"]),
        } for s in lst]
    return out


def check_quit(key):
    if key in (ord("q"), 27):
        raise Aborted("Oprit de utilizator.")


def draw_banner(frame, line1, line2):
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 95), (20, 20, 20), -1)
    cv2.putText(frame, line1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    cv2.putText(frame, line2, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def wait_for_hand(tracker, cam, person):
    hand_since = None
    log("Ridica mana in cadru (porneste automat in 4s sau apasa SPACE)...")
    while True:
        frame, result, now = tracker.process(cam)
        if result.hand_landmarks:
            hand_since = hand_since or now
            countdown = max(0.0, 4.0 - (now - hand_since))
            txt = f"[{person}] Mana detectata! Porneste in {countdown:.1f}s (sau SPACE)"
            if countdown <= 0:
                return
        else:
            hand_since = None
            txt = f"[{person}] Arata mana la camera pentru a incepe..."
        cv2.putText(frame, txt, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.imshow("Calibrare", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            return
        check_quit(key)


def prep_countdown(tracker, cam, instruction, person):
    t0 = time.monotonic()
    while time.monotonic() - t0 < PREP_SECONDS:
        frame, _, now = tracker.process(cam)
        left = PREP_SECONDS - (now - t0)
        draw_banner(frame, instruction, f"[{person}] Pregateste-te... start in {max(left, 0):.1f}s")
        cv2.imshow("Calibrare", frame)
        check_quit(cv2.waitKey(1) & 0xFF)


def run_phase(tracker, cam, key, instruction, duration, person, rearm, target):
    prep_countdown(tracker, cam, instruction, person)
    log(f"--> START {instruction}")
    feats = FeatureExtractor()
    samples = []
    total_frames = detected = 0
    live_count = 0
    phase_start = time.monotonic()

    while True:
        frame, result, now = tracker.process(cam)
        remaining = duration - (now - phase_start)
        if remaining <= 0:
            break
        h, w = frame.shape[:2]
        total_frames += 1

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            f = feats.update(lm, now)
            if now - phase_start >= TRIM_START:
                detected += 1
                samples.append(f)
                if key == "SWIPE" and detected % 8 == 0:
                    live_count = len(segment_swipes(samples, rearm))
            cv2.circle(frame, (int(f["cx"] * w), int(f["cy"] * h)), 8, (0, 255, 0), -1)
        else:
            cv2.putText(frame, "MANA PIERDUTA", (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        rate = detected / max(total_frames, 1)
        extra = f" | Swipe-uri: {live_count}/{target}" if target else ""
        draw_banner(frame, instruction,
                    f"[{person}] {remaining:.1f}s | Detectie: {rate * 100:.0f}% | Esantioane: {len(samples)}{extra}")
        done = 1 - remaining / duration
        cv2.rectangle(frame, (0, 90), (int(w * done), 95), (0, 255, 0), -1)
        cv2.imshow("Calibrare", frame)
        check_quit(cv2.waitKey(1) & 0xFF)

    rate = detected / max(total_frames, 1)
    log(f"    Gata {key}: {len(samples)} cadre, detectie {rate * 100:.0f}%.")
    return samples, rate


def evaluate_phase(key, samples, rate, rearm):
    problems = []
    if rate < MIN_DETECTION_RATE:
        problems.append(f"Mana a fost detectata doar in {rate * 100:.0f}% din timp (minim {MIN_DETECTION_RATE * 100:.0f}%). "
                        "Imbunatateste lumina / apropie-te / evita fundalul aglomerat.")
    if len(samples) < 50:
        problems.append("Prea putine esantioane.")
    if key == "SWIPE":
        n = len(segment_swipes(samples, rearm))
        if n < MIN_SWIPES:
            problems.append(f"Doar {n} swipe-uri detectate (minim {MIN_SWIPES}). Fa miscari mai ample si mai rapide, cu pauze.")
    if key in ("ZOOM_IN", "ZOOM_OUT") and samples:
        curled = sum(1 for s in samples if s["curled"]) / len(samples)
        if curled < 0.5:
            problems.append(f"Doar {curled * 100:.0f}% din cadre au degetele (mijlociu/inelar/mic) stranse. "
                            "Tine-le inchise si misca doar degetul mare si aratatorul.")
    return problems


def review_screen(key, problems):
    canvas = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
    cv2.putText(canvas, f"Faza {key}: calitate slaba", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    y = 150
    for p in problems:
        words, line = p.split(), ""
        for wd in words:
            if len(line) + len(wd) > 70:
                cv2.putText(canvas, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y += 34
                line = ""
            line += wd + " "
        cv2.putText(canvas, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 55
    cv2.putText(canvas, "R = repeta faza   |   SPACE = pastreaza oricum   |   Q = iesire",
                (40, CAM_HEIGHT - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.imshow("Calibrare", canvas)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("r"), ord("R")):
            return True
        if k == ord(" "):
            return False
        check_quit(k)


def collect_session(person_name):
    log("Se incarca modelul MediaPipe si camera...")
    landmarker = vision.HandLandmarker.create_from_options(options)
    cam = cv2.VideoCapture(0)
    try:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        if not cam.isOpened():
            raise Aborted("Camera nu a putut fi deschisa!")

        tracker = HandTracker(landmarker)
        wait_for_hand(tracker, cam, person_name)

        samples, rates = {}, {}
        rearm = 0.25
        for key, instruction, duration, target in PHASES:
            while True:
                s, rate = run_phase(tracker, cam, key, instruction, duration, person_name, rearm, target)
                problems = evaluate_phase(key, s, rate, rearm)
                if not problems or not review_screen(key, problems):
                    break
                log(f"    Repet faza {key}...")
            samples[key], rates[key] = s, rate
            if key == "IDLE":
                rearm = estimate_rearm(s)

        meta = {
            "data": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rezolutie": [int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))],
            "fps_camera": cam.get(cv2.CAP_PROP_FPS),
            "rata_detectie": {k: round(v, 3) for k, v in rates.items()},
        }
        return {"meta": meta, "samples": compact(samples, tracker.start)}
    finally:
        cam.release()
        cv2.destroyAllWindows()
        landmarker.close()


def choose_person_name(dataset):
    default_name = f"persoana_{len(dataset) + 1}"
    name = input(f"Numele persoanei pentru calibrare [{default_name}]: ").strip() or default_name
    if name in dataset:
        ans = input(f"'{name}' exista deja. Suprascrii? [d/N]: ").strip().lower()
        if ans != "d":
            i = 2
            while f"{name}_{i}" in dataset:
                i += 1
            name = f"{name}_{i}"
            log(f"Folosesc numele: {name}")
    return name


def recompute_only():
    dataset = load_dataset()
    if not dataset:
        log("Dataset gol sau inexistent.")
        return
    best = compute_multi_person_constants(dataset)
    save_json(CONFIG_PATH, best, indent=4)
    print_report(best, len(dataset), list(dataset.keys()))
    log(f"Config salvat in: {CONFIG_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Calibrare gesturi (swipe / zoom)")
    parser.add_argument("--recompute", action="store_true",
                        help="Recalculeaza constantele din dataset, fara camera")
    parser.add_argument("--remove", metavar="NUME", help="Sterge o persoana din dataset si recalculeaza")
    args = parser.parse_args()

    if args.remove:
        dataset = load_dataset()
        if dataset.pop(args.remove, None) is None:
            log(f"Persoana '{args.remove}' nu exista. Disponibile: {', '.join(dataset) or '-'}")
            return
        save_json(DATASET_PATH, dataset)
        log(f"Am sters '{args.remove}'.")
        recompute_only()
        return
    if args.recompute:
        recompute_only()
        return

    if not os.path.exists(MODEL_PATH):
        log(f"EROARE: Nu gasesc modelul la {MODEL_PATH}")
        return

    dataset = load_dataset()
    person_name = choose_person_name(dataset)
    log(f"Pregatire sesiune pentru: {person_name} (sesiuni existente in dataset: {len(dataset)})")

    try:
        entry = collect_session(person_name)
    except Aborted as e:
        log(str(e))
        return

    dataset[person_name] = entry
    save_json(DATASET_PATH, dataset)
    best = compute_multi_person_constants(dataset)
    save_json(CONFIG_PATH, best, indent=4)

    print_report(best, len(dataset), list(dataset.keys()))
    log(f"Dataset salvat in: {DATASET_PATH}")
    log(f"Config salvat in:  {CONFIG_PATH}")


if __name__ == "__main__":
    main()