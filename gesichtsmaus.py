"""
Gesichtsmaus v4: Mauszeiger per Kopf ODER per Augen steuern, Klicken/Scrollen mit zwei Fingern.

MODI (umschalten mit Strg+Alt+M):
  KOPF   - Kopf bewegen = Zeiger bewegen (schnell, stabil)
  AUGEN  - Blickrichtung steuert den Zeiger (braucht 9-Punkte-Kalibrierung, Webcam-genau = grob)

GESTEN (Hand mit der Handfläche zur Kamera, beide Modi):
  Zeige- + Mittelfinger ausgestreckt (✌️), Ring-/kleiner Finger eingeklappt:
      nach oben/unten wischen   -> Scrollen (nach unten wischen = nach unten scrollen)
      kurz tippen (Finger kurz nach vorn kippen) -> Klick
      zweimal tippen            -> Doppelklick
  Zeige-, Mittel- + Ringfinger ausgestreckt, kleiner Finger eingeklappt:
      kurz tippen               -> Rechtsklick

GLOBALE TASTENKÜRZEL (auch ohne Fenster-Fokus, nur Windows):
  Strg+Alt+M = Modus wechseln       Strg+Alt+P = Pause an/aus
  Strg+Alt+C = Neu kalibrieren      Strg+Alt+Q = Beenden
  Strg+Alt+Z = Augen-Modus nachjustieren: auf die Bildschirmmitte schauen und drücken
  (Kopf-Modus: Mitte neu setzen, gerade in die Kamera schauen / Augen-Modus: 15 Punkte)

Installation (Python 3.10 - 3.12):
  pip install opencv-python mediapipe==0.10.14 pyautogui
"""

import ctypes
import math
import sys
import threading
import time
import traceback
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# ======================= Einstellungen =======================
START_MODE = "head"         # "head" (Kopf) oder "eye" (Augen)
CAMERA_INDEX = 0
USE_DSHOW = True            # schnelleres Kamera-Backend unter Windows (bei Problemen: False)
HD_CAMERA = True            # True = 1280x720 (genauere Augen), False = 640x480 mit bis zu 60 FPS
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = (1280, 720, 30) if HD_CAMERA else (640, 480, 60)
PROC_WIDTH, PROC_HEIGHT = 320, 240               # Erkennung für Kopf + Hände (kleiner = schneller)
SHOW_PREVIEW = True         # False = kein Fenster (Beenden dann mit Strg+Alt+Q oder Strg+C)
PREVIEW_W, PREVIEW_H = 480, 360

# --- Kopf-Modus ---
GAIN_X = 3.5                # Empfindlichkeit horizontal (höher = weniger Kopfbewegung nötig)
GAIN_Y = 4.5                # Empfindlichkeit vertikal
FILTER_MIN_CUTOFF = 1.0     # kleiner = ruhiger bei langsamer Bewegung
FILTER_BETA = 0.015         # größer = schnelleres Mitgehen bei schneller Bewegung
CALIB_FRAMES = 30           # Bilder, über die die Kopf-Mitte gemittelt wird

# --- Augen-Modus ---
GAZE_SMOOTH_FRAMES = 9      # Median über so viele Bilder (größer = ruhiger, aber träger)
GAZE_START_RADIUS = 0.045   # Zeiger bewegt sich erst, wenn der Blick weiter weg ist (Anteil der Bildschirmbreite)
GAZE_STOP_RADIUS = 0.022    # ... und bleibt stehen, sobald er so nah dran ist
GAZE_GLIDE = 0.25           # wie schnell der Zeiger zum Blickpunkt gleitet (0..1)
GAZE_RIDGE_LIN = 1.0        # Regularisierung der Kalibrierung (Augen / Kopf / Kurven-Terme)
GAZE_RIDGE_HEAD = 5.0
GAZE_RIDGE_QUAD = 40.0
CAL_COLS, CAL_ROWS = 5, 3   # Kalibrierpunkte (5x3 = 15)
CAL_SETTLE = 0.8            # Sekunden, die du pro Punkt zum Hinschauen bekommst
CAL_COLLECT = 1.0           # Sekunden, in denen pro Punkt gemessen wird
CAL_MIN_SAMPLES = 150
BLINK_THRESHOLD = 0.15      # darunter gilt das Auge als geschlossen (Blinzeln wird ignoriert)

# --- Zwei-Finger-Gesten ---
DOUBLE_WINDOW = 0.45        # Sekunden, in denen ein zweites Tippen ein Doppelklick wird
ENABLE_RIGHT_CLICK = True   # Drei-Finger-Tippen = Rechtsklick
TAP_DIP_RATIO = 0.86        # Tippen erkannt, wenn die Finger auf diesen Anteil "einknicken"
TAP_RETURN_RATIO = 0.93     # ... und danach wieder über diesen Anteil kommen
TAP_MAX_TIME = 0.5          # so schnell muss das Tippen sein (Sekunden)
TAP_LOOKBACK = 0.12         # Klick landet dort, wo der Zeiger kurz VOR dem Tippen war
POSE_ENTER_FRAMES = 3       # Geste muss so viele Bilder stabil sein
POSE_GRACE = 0.5            # Pose darf kurz (beim Tippen) "verloren gehen"
SCROLL_GAIN = 3000          # Scroll-Stärke (höher = schneller)
SCROLL_MIN_SPEED = 0.30     # Mindest-Wischgeschwindigkeit (Bildhöhen/Sek.)
SCROLL_INVERT = False       # True = Handy-Stil (nach unten wischen = Seite nach oben)
SWAP_HANDS = False          # True, falls links/rechts vertauscht erkannt werden
# =============================================================

IS_WIN = sys.platform == "win32"
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
SCREEN_W, SCREEN_H = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
NOSE_IDX = (1, 4, 5, 195, 197)
R_EYE = dict(a=33, b=133, up=159, lo=145, iris=468)
L_EYE = dict(a=362, b=263, up=386, lo=374, iris=473)


# ---------------------------------------------------------------- Hilfsfunktionen
def setup_windows():
    if not IS_WIN:
        return
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    try:
        ctypes.windll.winmm.timeBeginPeriod(1)
    except Exception:
        pass
    try:
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00008000)  # ABOVE_NORMAL
    except Exception:
        pass


def teardown_windows():
    if IS_WIN:
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass


def move_cursor(x, y):
    if IS_WIN:
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
    else:
        pyautogui.moveTo(x, y)


def dist(a, b):
    """Abstand zweier Hand-Landmarks (Seitenverhältnis des kleinen Bildes eingerechnet)."""
    return math.hypot((a.x - b.x) * PROC_WIDTH, (a.y - b.y) * PROC_HEIGHT)


def finger_extended(lm, tip, pip):
    return dist(lm[tip], lm[0]) > dist(lm[pip], lm[0])


def hand_pose(lm):
    """2 = Zeige+Mittelfinger, 3 = Zeige+Mittel+Ringfinger, sonst None."""
    i = finger_extended(lm, 8, 6)
    m = finger_extended(lm, 12, 10)
    r = finger_extended(lm, 16, 14)
    p = finger_extended(lm, 20, 18)
    if i and m and not r and not p:
        return 2
    if i and m and r and not p:
        return 3
    return None


FINGER_PAIRS = {2: ((8, 5), (12, 9)), 3: ((8, 5), (12, 9), (16, 13))}


def extension(lm, pose):
    """Wie weit die Finger ausgestreckt wirken (Fingerspitze-Handgelenk / Knöchel-Handgelenk)."""
    vals = []
    for tip, mcp in FINGER_PAIRS[pose]:
        d0 = dist(lm[mcp], lm[0]) or 1e-6
        vals.append(dist(lm[tip], lm[0]) / d0)
    return sum(vals) / len(vals)


class OneEuro:
    """One-Euro-Filter: ruhig bei langsamer, schnell bei schneller Bewegung."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class GazeModel:
    """Bildet Augen- und Kopf-Merkmale per regularisierter Regression auf Bildschirmpunkte ab."""

    def __init__(self, mean, std, w, err, cv_err):
        self.mean, self.std, self.w, self.err, self.cv_err = mean, std, w, err, cv_err

    @staticmethod
    def feats(h, v, yaw, pitch):
        return np.array([h, v, yaw, pitch, h * v, h * h, v * v], dtype=float)

    @staticmethod
    def _solve(Z, T):
        reg = np.diag([0.0, GAZE_RIDGE_LIN, GAZE_RIDGE_LIN, GAZE_RIDGE_HEAD, GAZE_RIDGE_HEAD,
                       GAZE_RIDGE_QUAD, GAZE_RIDGE_QUAD, GAZE_RIDGE_QUAD])
        return np.linalg.solve(Z.T @ Z + reg, Z.T @ T)

    def predict(self, h, v, yaw, pitch):
        z = (self.feats(h, v, yaw, pitch) - self.mean) / self.std
        out = np.concatenate(([1.0], z)) @ self.w
        return float(out[0]), float(out[1])

    @classmethod
    def fit(cls, samples, targets, groups):
        X = np.array([cls.feats(*smp) for smp in samples])
        T = np.array(targets, dtype=float)
        G = np.array(groups)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        Z = np.hstack([np.ones((len(X), 1)), (X - mean) / std])

        w = cls._solve(Z, T)                                  # 1. Durchgang
        res = np.hypot(*(Z @ w - T).T)
        keep = res <= np.quantile(res, 0.8)                   # Ausreißer (Blinzeln, Abschweifen) raus
        w = cls._solve(Z[keep], T[keep])                      # 2. Durchgang
        pred = Z @ w
        err = float(np.mean(np.hypot(pred[:, 0] - T[:, 0], pred[:, 1] - T[:, 1])))

        cv = []                                               # ehrlicher Fehler: je Punkt weglassen
        for gid in np.unique(G):
            tr, te = (G != gid) & keep, G == gid
            wg = cls._solve(Z[tr], T[tr])
            p = Z[te] @ wg
            cv.append(np.median(np.hypot(p[:, 0] - T[te, 0], p[:, 1] - T[te, 1])))
        return cls(mean, std, w, err, float(np.mean(cv)))


class GazeSmoother:
    """Beruhigt den Blick: Median über mehrere Bilder + Zeiger bewegt sich nur bei echten Blickwechseln."""

    def __init__(self):
        self.bx = deque(maxlen=GAZE_SMOOTH_FRAMES)
        self.by = deque(maxlen=GAZE_SMOOTH_FRAMES)
        self.cur = None
        self.moving = False
        self.raw_med = None

    def reset(self):
        self.bx.clear()
        self.by.clear()
        self.cur = None
        self.moving = False
        self.raw_med = None

    def update(self, px, py):
        self.bx.append(px)
        self.by.append(py)
        mx, my = float(np.median(self.bx)), float(np.median(self.by))
        self.raw_med = (mx, my)
        if self.cur is None:
            self.cur = (mx, my)
            return self.cur
        cx, cy = self.cur
        d = math.hypot(mx - cx, my - cy)
        if not self.moving and d > GAZE_START_RADIUS * SCREEN_W:
            self.moving = True
        elif self.moving and d < GAZE_STOP_RADIUS * SCREEN_W:
            self.moving = False
        if self.moving:
            cx += GAZE_GLIDE * (mx - cx)
            cy += GAZE_GLIDE * (my - cy)
            self.cur = (cx, cy)
        return self.cur


def eye_feature(lm, e, W, H):
    """Iris-Position relativ zu den Augenwinkeln (drehinvariant). Rückgabe: h, v, Offenheit."""
    def P(i):
        return lm[i].x * W, lm[i].y * H

    p1, p2 = P(e["a"]), P(e["b"])
    ca, cb = (p1, p2) if p1[0] < p2[0] else (p2, p1)
    ux, uy = cb[0] - ca[0], cb[1] - ca[1]
    w = math.hypot(ux, uy)
    if w < 1e-6:
        return None
    ux, uy = ux / w, uy / w
    ir = P(e["iris"])
    h = ((ir[0] - ca[0]) * ux + (ir[1] - ca[1]) * uy) / w
    mx, my = (ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2
    v = ((ir[0] - mx) * (-uy) + (ir[1] - my) * ux) / w
    up, lo = P(e["up"]), P(e["lo"])
    openness = math.hypot(up[0] - lo[0], up[1] - lo[1]) / w
    return h, v, openness


def eye_features(lm, W, H):
    if len(lm) < 478:
        return None
    parts = []
    for e in (R_EYE, L_EYE):
        f = eye_feature(lm, e, W, H)
        if f is None:
            return None
        parts.append(f)
    if (parts[0][2] + parts[1][2]) / 2 < BLINK_THRESHOLD:
        return None   # Blinzeln
    return (parts[0][0] + parts[1][0]) / 2, (parts[0][1] + parts[1][1]) / 2


def head_features(lm):
    """Kopfdrehung (Yaw/Pitch-Näherung) aus Nase, Wangen, Stirn und Kinn."""
    xr, xl = lm[234].x, lm[454].x
    yaw = (lm[1].x - (xl + xr) / 2) / (abs(xl - xr) + 1e-6)
    top, bottom = lm[10].y, lm[152].y
    pitch = (lm[1].y - (top + bottom) / 2) / (abs(bottom - top) + 1e-6)
    return yaw, pitch


class State:
    """Gemeinsamer Zustand zwischen Kamera-, Gesichts-, Hand- und Anzeige-Thread."""

    def __init__(self):
        self.stop = False
        self.paused = False
        self.mode = START_MODE
        # Kopf-Modus
        self.recalibrate = True
        self.calibrating = False
        self.center = None
        # Augen-Modus
        self.gaze_model = None
        self.gaze_feat = None
        self.gaze_calibrating = False
        self.want_gaze_calib = (START_MODE == "eye")
        self.gaze_offset = (0.0, 0.0)
        self.zero_request = False
        # gemeinsam
        self.nose = None
        self.hand_draw = []
        self.freeze_until = 0.0
        self.scroll_until = 0.0
        self.history = deque(maxlen=300)    # (Zeit, x, y) der Zeigerposition
        self.tap_ratio = None
        self.pose_text = ""
        self.face_fps = 0.0
        self.hand_fps = 0.0


def pos_at(st, t):
    """Zeigerposition zum Zeitpunkt t (aus dem Verlauf) oder None."""
    for ts, x, y in reversed(list(st.history)):
        if ts <= t:
            return (x, y) if t - ts < 0.5 else None
    return None


def open_camera():
    """Probiert mehrere Backends, Kamera-Nummern und Auflösungen durch, bis wirklich ein Bild kommt."""
    if IS_WIN:
        backends = [("DirectShow", cv2.CAP_DSHOW), ("MSMF", getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)),
                    ("Standard", cv2.CAP_ANY)]
        if not USE_DSHOW:
            backends = backends[1:] + backends[:1]
    else:
        backends = [("Standard", cv2.CAP_ANY)]
    indices = [CAMERA_INDEX] + [i for i in range(4) if i != CAMERA_INDEX]
    profiles = [(CAM_WIDTH, CAM_HEIGHT, CAM_FPS, True), (640, 480, 30, False), (None, None, None, False)]

    for idx in indices:
        for bname, backend in backends:
            for w, h, fps, mjpg in profiles:
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    break                                   # dieses Backend/diese Nummer geht nicht
                if mjpg:
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if w:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    cap.set(cv2.CAP_PROP_FPS, fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok = False
                for _ in range(15):                         # Kameras brauchen kurz bis zum ersten Bild
                    ok, _frame = cap.read()
                    if ok:
                        break
                    time.sleep(0.1)
                if ok:
                    rw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    rh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"Kamera {idx} geöffnet ({bname}, {rw}x{rh})")
                    return cap
                cap.release()
    return None


class CameraStream:
    """Liest die Kamera in einem eigenen Thread (neuestes Bild, gespiegelt)."""

    def __init__(self):
        self.cap = open_camera()
        self.cond = threading.Condition()
        self.frame = None        # klein (für Kopf + Hände)
        self.frame_hi = None     # volle Auflösung (für die Augen)
        self.frame_id = 0
        self.running = self.cap is not None
        if self.running:
            threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                self.running = False
                break
            flipped = cv2.flip(f, 1)
            small = cv2.resize(flipped, (PROC_WIDTH, PROC_HEIGHT), interpolation=cv2.INTER_AREA)
            with self.cond:
                self.frame = small
                self.frame_hi = flipped
                self.frame_id += 1
                self.cond.notify_all()

    def wait_new(self, last_id, timeout=0.1):
        with self.cond:
            if self.frame_id == last_id:
                self.cond.wait(timeout)
            if self.frame_id != last_id and self.frame is not None:
                return self.frame, self.frame_id
        return None

    def stop(self):
        self.running = False
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()


def safe(fn, st):
    def wrapper(*args):
        try:
            fn(*args)
        except Exception:
            traceback.print_exc()
            st.stop = True
    return wrapper


# ---------------------------------------------------------------- Gesichts-Thread (Kopf + Augen)
def make_face(refine):
    return mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=refine,
                                           min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)


def face_worker(cam, st):
    mode = st.mode
    face = make_face(mode == "eye")
    fh = (OneEuro(FILTER_MIN_CUTOFF, FILTER_BETA), OneEuro(FILTER_MIN_CUTOFF, FILTER_BETA))
    smoother = GazeSmoother()
    last_moved = None
    calib = []
    last = -1
    fps_n, fps_t = 0, time.perf_counter()

    while not st.stop:
        if st.mode != mode:                       # Modus wurde gewechselt
            mode = st.mode
            face.close()
            face = make_face(mode == "eye")       # Augen-Modus braucht das Iris-Modell
            for f in fh:
                f.reset()
            smoother.reset()
            last_moved = None
            calib = []

        got = cam.wait_new(last, 0.1)
        if got is None:
            continue
        frame, last = got
        if mode == "eye" and cam.frame_hi is not None:
            frame = cam.frame_hi                  # volle Auflösung für genaue Iris-Position
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = face.process(rgb)
        now = time.perf_counter()

        fps_n += 1
        if now - fps_t >= 1.0:
            st.face_fps = fps_n / (now - fps_t)
            fps_n, fps_t = 0, now

        if not res.multi_face_landmarks:
            st.nose = None
            continue
        lm = res.multi_face_landmarks[0].landmark
        nose = (sum(lm[i].x for i in NOSE_IDX) / len(NOSE_IDX),
                sum(lm[i].y for i in NOSE_IDX) / len(NOSE_IDX))
        st.nose = nose

        if mode == "head":
            # --- Kalibrierung der Mitte ---
            if st.recalibrate:
                st.recalibrate = False
                st.calibrating = True
                calib = []
            if st.calibrating:
                calib.append(nose)
                if len(calib) >= CALIB_FRAMES:
                    st.center = (sum(p[0] for p in calib) / len(calib),
                                 sum(p[1] for p in calib) / len(calib))
                    st.calibrating = False
                    fh[0].reset()
                    fh[1].reset()
                continue
            if st.center is None or st.paused:
                continue
            tx = SCREEN_W / 2 + (nose[0] - st.center[0]) * GAIN_X * SCREEN_W
            ty = SCREEN_H / 2 + (nose[1] - st.center[1]) * GAIN_Y * SCREEN_H
            tx = min(max(tx, 0), SCREEN_W - 1)
            ty = min(max(ty, 0), SCREEN_H - 1)
            x, y = fh[0](tx, now), fh[1](ty, now)
            st.history.append((now, x, y))
            if now < st.freeze_until or now < st.scroll_until:
                continue
            move_cursor(x, y)

        else:
            # --- Augen-Modus ---
            feat = eye_features(lm, frame.shape[1], frame.shape[0])
            if feat is None:                      # Blinzeln o.ä.: Zeiger bleibt, wo er ist
                continue
            yaw, pitch = head_features(lm)
            st.gaze_feat = (now, feat[0], feat[1], yaw, pitch)
            model = st.gaze_model
            if model is None or st.gaze_calibrating or st.paused:
                continue
            px, py = model.predict(feat[0], feat[1], yaw, pitch)
            cx, cy = smoother.update(px, py)
            if st.zero_request:                   # Nachjustieren: du schaust gerade auf die Bildschirmmitte
                st.zero_request = False
                mx, my = smoother.raw_med
                st.gaze_offset = (SCREEN_W / 2 - mx, SCREEN_H / 2 - my)
                print("Nachjustiert.")
            x = min(max(cx + st.gaze_offset[0], 0), SCREEN_W - 1)
            y = min(max(cy + st.gaze_offset[1], 0), SCREEN_H - 1)
            st.history.append((now, x, y))
            if now < st.freeze_until or now < st.scroll_until:
                continue
            if last_moved is None or abs(x - last_moved[0]) + abs(y - last_moved[1]) >= 1.0:
                move_cursor(x, y)
                last_moved = (x, y)


# ---------------------------------------------------------------- Gesten (Zwei-Finger)
class Gestures:
    def __init__(self, st):
        self.st = st
        self.taps = 0
        self.last_tap = 0.0
        self.tap_pos = None
        self.last_tap_evt = 0.0
        self.block_scroll_until = 0.0
        self.reset_pose()

    def reset_pose(self):
        self.pose = None
        self.label = None
        self.last_pose_t = 0.0
        self.pose_frames = 0
        self.baseline = None
        self.dipping = False
        self.dip_t = 0.0
        self.armed = True
        self.prev_y = self.y_s = None
        self.prev_t = 0.0
        self.accum = 0.0
        self.st.tap_ratio = None
        self.st.pose_text = ""

    def reset_all(self):
        self.reset_pose()
        self.taps = 0

    def update(self, hands, now):
        st = self.st

        # 1) Pose im aktuellen Bild suchen (rechte Hand bevorzugt)
        raw = None
        for label in ("R", "L"):
            lm = hands.get(label)
            if lm is not None:
                p = hand_pose(lm)
                if p:
                    raw = (label, p, lm)
                    break
        hand = None
        if raw:
            label, pose, hand = raw
            if pose != self.pose or label != self.label:
                self.reset_pose()
                self.pose, self.label = pose, label
            self.last_pose_t = now
            self.pose_frames += 1
        elif self.pose is not None and now - self.last_pose_t <= POSE_GRACE:
            hand = hands.get(self.label)           # beim Tippen darf die Pose kurz wackeln
        elif self.pose is not None:
            self.reset_pose()

        stable = self.pose_frames >= POSE_ENTER_FRAMES
        ratio = None
        if self.pose is not None and hand is not None:
            ext = extension(hand, self.pose)
            if self.baseline is None:
                self.baseline = ext
            ratio = ext / self.baseline
            if raw and not self.dipping:           # Grundwert: schnell hoch, langsam runter
                self.baseline += (0.3 if ext > self.baseline else 0.02) * (ext - self.baseline)
        st.tap_ratio = ratio
        st.pose_text = {2: "2 Finger", 3: "3 Finger"}.get(self.pose, "")

        # 2) Tippen erkennen: Finger knicken kurz ein und kommen zurück
        if stable and ratio is not None:
            if not self.dipping:
                if ratio >= TAP_RETURN_RATIO:
                    self.armed = True
                if self.armed and ratio < TAP_DIP_RATIO and now >= st.scroll_until:
                    self.dipping = True
                    self.armed = False
                    self.dip_t = now
                    st.freeze_until = max(st.freeze_until, now + 0.6)
            else:
                if raw and ratio >= TAP_RETURN_RATIO:
                    self.dipping = False
                    self.armed = True
                    if now - self.dip_t <= TAP_MAX_TIME and now - self.last_tap_evt > 0.12:
                        self.last_tap_evt = now
                        self.fire_tap(now)
                elif now - self.dip_t > TAP_MAX_TIME:
                    self.dipping = False           # war nur Entspannen der Hand, kein Tippen

        # 3) Scrollen: Zwei Finger nach oben/unten wischen
        can_scroll = (self.pose == 2 and raw is not None and stable and not self.dipping
                      and ratio is not None and ratio >= TAP_RETURN_RATIO
                      and now >= self.block_scroll_until)
        if can_scroll:
            y = sum(hand[i].y for i in (0, 5, 9, 13, 17)) / 5   # Handmitte: Finger-Einknicken zählt nicht
            self.y_s = y if self.y_s is None else self.y_s + 0.6 * (y - self.y_s)
            if self.prev_y is not None:
                dt = max(now - self.prev_t, 1e-3)
                dy = self.y_s - self.prev_y
                if abs(dy) / dt > SCROLL_MIN_SPEED:
                    self.accum += (dy if SCROLL_INVERT else -dy) * SCROLL_GAIN
                    amount = int(self.accum)
                    if amount:
                        pyautogui.scroll(amount)
                        self.accum -= amount
                    st.scroll_until = now + 0.4
            self.prev_y, self.prev_t = self.y_s, now
        else:
            self.prev_y = self.y_s = None
            self.accum = 0.0

    def fire_tap(self, now):
        st = self.st
        pos = pos_at(st, self.dip_t - TAP_LOOKBACK)
        if pos is None:
            pos = pyautogui.position()
        x, y = int(pos[0]), int(pos[1])
        self.block_scroll_until = now + 0.4
        if self.pose == 3:
            if ENABLE_RIGHT_CLICK:
                pyautogui.rightClick(x, y)
                self.taps = 0
                st.freeze_until = now + 0.25
            return
        self.taps += 1
        self.last_tap = now
        if self.taps == 1:
            self.tap_pos = (x, y)
            st.freeze_until = now + DOUBLE_WINDOW + 0.1
        else:
            pyautogui.doubleClick(*self.tap_pos)
            self.taps = 0
            st.freeze_until = now + 0.25

    def pending(self, now):
        """Einzelklick auslösen, wenn nach dem ersten Tippen kein zweites kam."""
        if self.taps == 1 and now - self.last_tap > DOUBLE_WINDOW:
            pyautogui.click(*self.tap_pos)
            self.taps = 0
            self.st.freeze_until = now + 0.25


def hand_worker(cam, st):
    hands_model = mp_hands.Hands(max_num_hands=2, model_complexity=0,
                                 min_detection_confidence=0.6, min_tracking_confidence=0.6)
    gest = Gestures(st)
    last = -1
    fps_n, fps_t = 0, time.perf_counter()

    while not st.stop:
        got = cam.wait_new(last, 0.1)
        if got is None:
            continue
        frame, last = got
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = hands_model.process(rgb)
        now = time.perf_counter()

        fps_n += 1
        if now - fps_t >= 1.0:
            st.hand_fps = fps_n / (now - fps_t)
            fps_n, fps_t = 0, now

        hands = {"R": None, "L": None}
        draw = []
        if res.multi_hand_landmarks and res.multi_handedness:
            for lms, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                label = hd.classification[0].label            # Bild ist gespiegelt -> stimmt für den Nutzer
                if SWAP_HANDS:
                    label = "Right" if label == "Left" else "Left"
                hands["R" if label == "Right" else "L"] = lms.landmark
                draw.append(lms)
        st.hand_draw = draw

        if st.paused or st.gaze_calibrating:
            gest.reset_all()
            continue
        gest.update(hands, now)
        gest.pending(now)


# ---------------------------------------------------------------- Augen-Kalibrierung (Hauptthread)
def calibration_points():
    xs = np.linspace(0.07, 0.93, CAL_COLS)
    ys = np.linspace(0.10, 0.90, CAL_ROWS)
    pts = []
    for r, y in enumerate(ys):
        row = xs if r % 2 == 0 else xs[::-1]      # Schlangenlinie: kurze Wege für die Augen
        pts += [(float(x), float(y)) for x in row]
    return pts


def run_gaze_calibration(st):
    pts = calibration_points()
    win = "Augen-Kalibrierung"
    base = np.zeros((SCREEN_H, SCREEN_W, 3), np.uint8)
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    st.gaze_calibrating = True
    samples, targets, groups = [], [], []
    cancelled = False
    print(f"Augen-Kalibrierung: schau nacheinander auf die {len(pts)} Punkte (Esc = Abbrechen)")
    try:
        for idx, (fx_, fy_) in enumerate(pts):
            tx, ty = int(fx_ * SCREEN_W), int(fy_ * SCREEN_H)
            t0 = time.perf_counter()
            last_ts = None
            while not st.stop:
                el = time.perf_counter() - t0
                if el >= CAL_SETTLE + CAL_COLLECT:
                    break
                img = base.copy()
                r = int(30 - 20 * min(el / CAL_SETTLE, 1.0))
                cv2.circle(img, (tx, ty), r, (0, 255, 255), 2)
                cv2.circle(img, (tx, ty), 4, (0, 255, 255), -1)
                cv2.putText(img, f"Punkt {idx + 1}/{len(pts)}: einfach natuerlich ansehen - der Kopf darf leicht mitgehen (Esc = Abbrechen)",
                            (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow(win, img)
                if cv2.waitKey(1) & 0xFF == 27:
                    cancelled = True
                    break
                if el >= CAL_SETTLE:
                    g = st.gaze_feat
                    if g is not None and g[0] != last_ts and time.perf_counter() - g[0] < 0.3:
                        last_ts = g[0]
                        samples.append(g[1:5])
                        targets.append((tx, ty))
                        groups.append(idx)
            if cancelled or st.stop:
                break

        if cancelled or st.stop:
            print("Kalibrierung abgebrochen.")
            return
        if len(samples) < CAL_MIN_SAMPLES:
            print(f"Zu wenige Messwerte ({len(samples)}). Gesicht gut beleuchten, nicht zu weit weg, nochmal versuchen (Strg+Alt+C).")
            return

        model = GazeModel.fit(samples, targets, groups)
        st.gaze_model = model
        st.gaze_offset = (0.0, 0.0)
        print(f"Kalibrierung fertig. Geschätzter Fehler: ca. {model.cv_err:.0f} px "
              f"({model.cv_err / SCREEN_W * 100:.0f}% der Bildschirmbreite).")
        if model.cv_err > 0.2 * SCREEN_W:
            print("Hinweis: Fehler ist groß - mehr Licht von vorn, näher an die Kamera, neu kalibrieren.")

        # ---- Testphase: grünes Kreuz = wohin das Programm denkt, dass du schaust ----
        sm = GazeSmoother()
        t0 = time.perf_counter()
        last_ts = None
        while not st.stop and time.perf_counter() - t0 < 8.0:
            img = base.copy()
            for fx_, fy_ in pts:
                cv2.circle(img, (int(fx_ * SCREEN_W), int(fy_ * SCREEN_H)), 6, (70, 70, 70), -1)
            g = st.gaze_feat
            if g is not None and g[0] != last_ts and time.perf_counter() - g[0] < 0.3:
                last_ts = g[0]
                sm.update(*model.predict(*g[1:5]))
            if sm.cur is not None:
                cx = int(min(max(sm.cur[0], 0), SCREEN_W - 1))
                cy = int(min(max(sm.cur[1], 0), SCREEN_H - 1))
                cv2.drawMarker(img, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 40, 3)
            cv2.putText(img, f"Test: das gruene Kreuz zeigt, wohin du schaust (Fehler ca. {model.cv_err:.0f}px). Esc = weiter",
                        (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(win, img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv2.destroyWindow(win)
        st.gaze_calibrating = False


# ---------------------------------------------------------------- Hauptprogramm
class Hotkeys:
    """Globale Tastenkürzel Strg+Alt+M/P/C/Q/Z über GetAsyncKeyState (nur Windows)."""
    KEYS = {"mode": 0x4D, "pause": 0x50, "calib": 0x43, "quit": 0x51, "zero": 0x5A}

    def __init__(self):
        self.prev = {k: False for k in self.KEYS}

    def poll(self):
        if not IS_WIN:
            return []
        u32 = ctypes.windll.user32
        down = lambda vk: bool(u32.GetAsyncKeyState(vk) & 0x8000)
        mods = down(0x11) and down(0x12)
        events = []
        for name, vk in self.KEYS.items():
            now = mods and down(vk)
            if now and not self.prev[name]:
                events.append(name)
            self.prev[name] = now
        return events


def draw_preview(cam, st):
    frame = cam.frame
    if frame is None:
        return None
    img = cv2.resize(frame, (PREVIEW_W, PREVIEW_H))
    if st.nose is not None:
        cv2.circle(img, (int(st.nose[0] * PREVIEW_W), int(st.nose[1] * PREVIEW_H)), 5, (0, 255, 0), -1)
    for lms in st.hand_draw:
        mp_draw.draw_landmarks(img, lms, mp_hands.HAND_CONNECTIONS)

    mode_txt = "KOPF" if st.mode == "head" else "AUGEN"
    if st.mode == "head" and (st.center is None or st.calibrating):
        status = "KALIBRIERE - gerade in die Kamera schauen"
    elif st.mode == "eye" and st.gaze_model is None:
        status = "noch nicht kalibriert (Strg+Alt+C)"
    elif st.paused:
        status = "PAUSE"
    elif time.perf_counter() < st.scroll_until:
        status = "SCROLL"
    else:
        status = "AKTIV"
    if st.mode == "eye" and st.gaze_model is not None:
        status += f" (Fehler ~{st.gaze_model.cv_err:.0f}px)"
    cv2.putText(img, f"{mode_txt}: {status}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(img, f"Kopf/Auge {st.face_fps:.0f} FPS | Hand {st.hand_fps:.0f} FPS", (8, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if st.pose_text:
        r = f"{st.tap_ratio:.2f}" if st.tap_ratio is not None else "-"
        cv2.putText(img, f"{st.pose_text} | Tipp-Wert {r} (Tippen < {TAP_DIP_RATIO})", (8, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    cv2.putText(img, "Strg+Alt: M=Modus P=Pause C=Kalibr. Z=Mitte Q=Ende", (8, PREVIEW_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


def handle(action, st):
    if action == "mode":
        st.mode = "eye" if st.mode == "head" else "head"
        if st.mode == "head":
            st.recalibrate = True
            print("Modus: KOPF - bitte gerade in die Kamera schauen")
        else:
            print("Modus: AUGEN")
            if st.gaze_model is None:
                st.want_gaze_calib = True
    elif action == "pause":
        st.paused = not st.paused
        print("Pause" if st.paused else "Aktiv")
    elif action == "calib":
        if st.mode == "eye":
            st.want_gaze_calib = True
        else:
            st.recalibrate = True
            print("Kalibriere neu - bitte gerade in die Kamera schauen")
    elif action == "zero":
        if st.mode == "eye" and st.gaze_model is not None:
            st.zero_request = True
        else:
            print("Nachjustieren geht nur im Augen-Modus nach der Kalibrierung.")
    elif action == "quit":
        st.stop = True


def main():
    setup_windows()
    cam = CameraStream()
    if not cam.running:
        print("Keine Kamera gefunden. Wird sie gerade von einem anderen Programm benutzt (Teams, Browser, eine zweite Gesichtsmaus)? Außerdem prüfen: Windows-Einstellungen > Datenschutz > Kamera > \"Desktop-Apps Zugriff erlauben\".")
        teardown_windows()
        return

    st = State()
    threads = [
        threading.Thread(target=safe(face_worker, st), args=(cam, st), daemon=True),
        threading.Thread(target=safe(hand_worker, st), args=(cam, st), daemon=True),
    ]
    for t in threads:
        t.start()

    hot = Hotkeys()
    try:
        while not st.stop and cam.running:
            for action in hot.poll():
                handle(action, st)
            if st.want_gaze_calib:
                st.want_gaze_calib = False
                run_gaze_calibration(st)
            if SHOW_PREVIEW:
                img = draw_preview(cam, st)
                if img is not None:
                    cv2.imshow("Gesichtsmaus", img)
                key = cv2.waitKey(15) & 0xFF
                if key == ord("q"):
                    handle("quit", st)
                elif key == ord("p"):
                    handle("pause", st)
                elif key == ord("c"):
                    handle("calib", st)
                elif key == ord("m"):
                    handle("mode", st)
            else:
                time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        st.stop = True
        for t in threads:
            t.join(timeout=1.0)
        cam.stop()
        cv2.destroyAllWindows()
        teardown_windows()


if __name__ == "__main__":
    main()
