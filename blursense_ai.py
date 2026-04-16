"""
╔══════════════════════════════════════════════════════════════╗
║          BlurSense AI – Privacy-Aware CCTV System           ║
║        Real-time Face Detection with Privacy Control        ║
╚══════════════════════════════════════════════════════════════╝

Author  : BlurSense AI
Version : 1.0.0
Requires: opencv-python, requests

Install dependencies:
    pip install opencv-python requests

Run:
    python blursense_ai.py

Keyboard Controls:
    A → Authorized mode   (no blur)
    U → Unauthorized mode (blur faces + alerts)
    Q → Quit
"""

import cv2
import os
import time
import logging
import requests
from datetime import datetime


# ─────────────────────────────────────────────────────────────
#  CONFIGURATION – Edit these values to suit your setup
# ─────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "YOUR TOKEN"       # e.g. "123456:ABC-DEF..."
TELEGRAM_CHAT_ID   = "ID"         # e.g. "987654321"

SNAPSHOT_DIR       = "snapshots"                  # Folder to save alert snapshots
ALERT_COOLDOWN_SEC = 10                           # Minimum seconds between alerts
MIN_FACES_TO_ALERT = 1                            # Alert only when ≥ N faces found

# Haar Cascade – OpenCV ships this file; path resolved automatically below
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Detection tuning
SCALE_FACTOR  = 1.1    # How much the image size is reduced at each image scale
MIN_NEIGHBORS = 5      # Higher = fewer, more-accurate detections
MIN_FACE_SIZE = (60, 60)

# Display colours  (BGR)
COLOR_AUTH   = (0, 220, 80)    # Green  – Authorized
COLOR_UNAUTH = (0, 60, 230)    # Red    – Unauthorized
COLOR_BOX    = (255, 200, 0)   # Cyan   – Bounding box
COLOR_WHITE  = (255, 255, 255)

# Blur intensity (must be odd)
BLUR_KERNEL = (51, 51)


# ─────────────────────────────────────────────────────────────
#  LOGGING SETUP
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                       # Console
        logging.FileHandler("blursense.log", mode="a") # Log file
    ]
)
logger = logging.getLogger("BlurSenseAI")


# ═══════════════════════════════════════════════════════════════
#  CLASS 1 – FaceDetector
#  Wraps Haar Cascade logic: detect faces, draw boxes, blur
# ═══════════════════════════════════════════════════════════════
class FaceDetector:
    """Handles face detection and privacy-related frame processing."""

    def __init__(self, cascade_path: str):
        """
        Load the Haar Cascade classifier.

        Args:
            cascade_path: Full path to haarcascade_frontalface_default.xml
        """
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(
                f"Haar Cascade file not found: {cascade_path}\n"
                "Reinstall opencv-python:  pip install --upgrade opencv-python"
            )
        self.classifier = cv2.CascadeClassifier(cascade_path)
        logger.info("FaceDetector initialised with cascade: %s", cascade_path)

    # ── Core detection ──────────────────────────────────────
    def detect_faces(self, gray_frame):
        """
        Detect faces in a grayscale frame.

        Args:
            gray_frame: Single-channel (grayscale) OpenCV image.

        Returns:
            List of (x, y, w, h) rectangles for each detected face.
        """
        faces = self.classifier.detectMultiScale(
            gray_frame,
            scaleFactor=SCALE_FACTOR,
            minNeighbors=MIN_NEIGHBORS,
            minSize=MIN_FACE_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        # detectMultiScale returns an empty tuple when nothing found
        return faces if len(faces) > 0 else []

    # ── Blur a single region of interest ────────────────────
    @staticmethod
    def blur_region(frame, x: int, y: int, w: int, h: int):
        """
        Apply Gaussian blur to a rectangular region in-place.

        Args:
            frame: BGR frame (modified in place).
            x, y, w, h: Bounding-box coordinates.
        """
        roi = frame[y:y + h, x:x + w]
        blurred_roi = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)
        frame[y:y + h, x:x + w] = blurred_roi

    # ── Draw bounding box ────────────────────────────────────
    @staticmethod
    def draw_box(frame, x: int, y: int, w: int, h: int, color=COLOR_BOX):
        """Draw a rectangle around a detected face."""
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # ── Process complete frame ───────────────────────────────
    def process_frame(self, frame, unauthorized_mode: bool):
        """
        Detect all faces in `frame`, draw boxes, and optionally blur them.

        Args:
            frame: BGR video frame.
            unauthorized_mode: If True, faces are blurred.

        Returns:
            (annotated_frame, face_count)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray)

        for (x, y, w, h) in faces:
            if unauthorized_mode:
                self.blur_region(frame, x, y, w, h)
            self.draw_box(frame, x, y, w, h)

        return frame, len(faces)


# ═══════════════════════════════════════════════════════════════
#  CLASS 2 – AlertManager
#  Handles snapshots, Telegram messages, and cooldown logic
# ═══════════════════════════════════════════════════════════════
class AlertManager:
    """Manages alert dispatching: snapshots, Telegram, and cooldown."""

    def __init__(self, snapshot_dir: str, cooldown_sec: int):
        """
        Args:
            snapshot_dir: Directory path where snapshot images are saved.
            cooldown_sec: Minimum seconds that must pass between two alerts.
        """
        self.snapshot_dir = snapshot_dir
        self.cooldown_sec = cooldown_sec
        self._last_alert_time = 0.0   # epoch seconds of last alert

        os.makedirs(snapshot_dir, exist_ok=True)
        logger.info(
            "AlertManager ready  |  snapshots → '%s'  |  cooldown → %ds",
            snapshot_dir, cooldown_sec
        )

    # ── Cooldown guard ───────────────────────────────────────
    def _cooldown_passed(self) -> bool:
        """Return True if enough time has elapsed since the last alert."""
        return (time.time() - self._last_alert_time) >= self.cooldown_sec

    def _mark_alert(self):
        """Record that an alert was just dispatched."""
        self._last_alert_time = time.time()

    # ── Snapshot ─────────────────────────────────────────────
    def save_snapshot(self, frame) -> str:
        """
        Save the current frame as a JPEG snapshot.

        Args:
            frame: BGR image to save.

        Returns:
            Full path to the saved file (or empty string on failure).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = os.path.join(self.snapshot_dir, f"alert_{timestamp}.jpg")
        try:
            cv2.imwrite(filename, frame)
            logger.info("Snapshot saved → %s", filename)
            return filename
        except Exception as exc:
            logger.error("Failed to save snapshot: %s", exc)
            return ""

    # ── Telegram alert ───────────────────────────────────────
    def send_telegram_alert(self, face_count: int, timestamp: str) -> bool:
        """
        Send a Telegram alert via the Bot API.

        To activate:
          1. Create a bot with @BotFather → copy the token.
          2. Get your chat ID (send a message to your bot, then visit
             https://api.telegram.org/bot<TOKEN>/getUpdates).
          3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID at the top of this file.

        Args:
            face_count: Number of faces detected.
            timestamp:  Human-readable time string.

        Returns:
            True if the message was delivered successfully, False otherwise.
        """
        if TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
            logger.warning(
                "Telegram not configured – set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID."
            )
            return False

        message = (
            "🚨 *BlurSense AI ALERT* 🚨\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "🔴 *Unauthorized Access Detected*\n"
            f"👤 *Faces Detected:* `{face_count}`\n"
            f"🕐 *Time:* `{timestamp}`\n"
            "━━━━━━━━━━━━━━━━━━━━━━\n"
            "_BlurSense AI is monitoring your space._"
        )

        url     = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id"    : TELEGRAM_CHAT_ID,
            "text"       : message,
            "parse_mode" : "Markdown"
        }

        try:
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                logger.info("Telegram alert sent  |  faces=%d", face_count)
                return True
            else:
                logger.error(
                    "Telegram API error %d: %s",
                    response.status_code, response.text
                )
                return False
        except requests.exceptions.ConnectionError:
            logger.error("Telegram alert failed – no internet connection.")
            return False
        except requests.exceptions.Timeout:
            logger.error("Telegram alert failed – request timed out.")
            return False
        except Exception as exc:
            logger.error("Telegram alert unexpected error: %s", exc)
            return False

    # ── Master trigger ───────────────────────────────────────
    def trigger_alert(self, frame, face_count: int):
        """
        Trigger a full alert cycle (snapshot + Telegram) if cooldown allows.

        Args:
            frame:      Current BGR video frame (to snapshot).
            face_count: Number of faces currently detected.
        """
        if face_count < MIN_FACES_TO_ALERT or not self._cooldown_passed():
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.warning(
            "ALERT  |  UNAUTHORIZED mode  |  faces=%d  |  time=%s",
            face_count, timestamp
        )

        self.save_snapshot(frame)
        self.send_telegram_alert(face_count, timestamp)
        self._mark_alert()


# ═══════════════════════════════════════════════════════════════
#  CLASS 3 – BlurSenseApp
#  Main application: webcam loop, keyboard handling, UI overlay
# ═══════════════════════════════════════════════════════════════
class BlurSenseApp:
    """
    Top-level application controller.

    Orchestrates FaceDetector and AlertManager, manages the webcam
    capture loop, renders the HUD overlay, and handles keyboard input.
    """

    # Mode constants
    MODE_AUTHORIZED   = "AUTHORIZED"
    MODE_UNAUTHORIZED = "UNAUTHORIZED"

    def __init__(self):
        """Initialise sub-systems and prepare the application."""
        self.detector = FaceDetector(CASCADE_PATH)
        self.alerter  = AlertManager(SNAPSHOT_DIR, ALERT_COOLDOWN_SEC)

        self.mode      = self.MODE_AUTHORIZED   # Default to safe mode
        self.cap       = None                   # cv2.VideoCapture handle
        self.running   = False

        # FPS tracking
        self._fps_start = time.time()
        self._fps_count = 0
        self._fps       = 0.0

        logger.info("BlurSenseApp initialised  |  starting mode → %s", self.mode)

    # ── Webcam helpers ───────────────────────────────────────
    def _open_camera(self, index: int = 0):
        """Open the webcam. Raises RuntimeError on failure."""
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open webcam at index {index}. "
                "Check that no other application is using the camera."
            )
        # Suggest a comfortable capture resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info("Webcam opened  |  index=%d", index)

    def _release_camera(self):
        """Release the webcam and destroy all OpenCV windows."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Goodbye!")

    # ── Mode switching ───────────────────────────────────────
    def _set_mode(self, new_mode: str):
        """Switch operating mode and log the event."""
        if new_mode != self.mode:
            self.mode = new_mode
            logger.info("Mode switched → %s", self.mode)

    # ── FPS calculation ──────────────────────────────────────
    def _update_fps(self):
        """Recalculate FPS once per second."""
        self._fps_count += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self._fps       = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_start = time.time()

    # ── HUD overlay ──────────────────────────────────────────
    def _draw_hud(self, frame, face_count: int):
        """
        Render the heads-up display on top of the video frame.

        Displays: mode badge, face count, FPS, and keyboard hints.
        """
        h, w = frame.shape[:2]
        is_auth = (self.mode == self.MODE_AUTHORIZED)

        # ── Top banner ──────────────────────────────────────
        banner_color = COLOR_AUTH if is_auth else COLOR_UNAUTH
        cv2.rectangle(frame, (0, 0), (w, 52), banner_color, -1)  # filled rect

        # Mode label
        mode_label = f"  {'✔' if is_auth else '✘'}  {self.mode} MODE"
        cv2.putText(
            frame, mode_label,
            (10, 36),
            cv2.FONT_HERSHEY_DUPLEX, 1.0,
            COLOR_WHITE, 2, cv2.LINE_AA
        )

        # App name (top-right)
        app_label = "BlurSense AI"
        (tw, _), _ = cv2.getTextSize(app_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(
            frame, app_label,
            (w - tw - 10, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLOR_WHITE, 1, cv2.LINE_AA
        )

        # ── Info pills (bottom-left) ────────────────────────
        info_y = h - 60
        cv2.rectangle(frame, (0, info_y - 4), (300, h), (20, 20, 20), -1)

        face_text = f"Faces : {face_count}"
        fps_text  = f"FPS   : {self._fps:.1f}"
        cv2.putText(frame, face_text, (10, info_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, fps_text,  (10, info_y + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 1, cv2.LINE_AA)

        # ── Keyboard hint strip (bottom-right) ──────────────
        hints = "[A] Auth  [U] Unauth  [Q] Quit"
        (hw, _), _ = cv2.getTextSize(hints, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        hx = w - hw - 10
        cv2.rectangle(frame, (hx - 6, h - 30), (w, h), (20, 20, 20), -1)
        cv2.putText(
            frame, hints,
            (hx, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (180, 180, 180), 1, cv2.LINE_AA
        )

        # ── Flashing ALERT badge in unauthorized + faces ─────
        if not is_auth and face_count >= MIN_FACES_TO_ALERT:
            blink = int(time.time() * 2) % 2   # blink twice/sec
            if blink:
                alert_text = "⚠ ALERT: UNAUTHORIZED FACE DETECTED"
                (aw, _), _ = cv2.getTextSize(
                    alert_text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2
                )
                ax = (w - aw) // 2
                cv2.putText(
                    frame, alert_text,
                    (ax, h - 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    COLOR_UNAUTH, 2, cv2.LINE_AA
                )

    # ── Keyboard input handler ───────────────────────────────
    def _handle_key(self, key: int) -> bool:
        """
        Process a key press.

        Returns:
            False if the application should quit, True otherwise.
        """
        if key == -1:       # No key pressed this frame
            return True

        ch = chr(key & 0xFF).upper()

        if ch == 'Q':
            logger.info("Quit key pressed.")
            return False
        elif ch == 'A':
            self._set_mode(self.MODE_AUTHORIZED)
        elif ch == 'U':
            self._set_mode(self.MODE_UNAUTHORIZED)

        return True

    # ── Main loop ────────────────────────────────────────────
    def run(self):
        """
        Start the BlurSense AI application.

        Opens the webcam, enters the frame-processing loop, and cleans
        up on exit.
        """
        logger.info("=" * 60)
        logger.info("  BlurSense AI  –  Privacy-Aware CCTV System  STARTED")
        logger.info("=" * 60)

        try:
            self._open_camera(index=0)
        except RuntimeError as exc:
            logger.critical("Startup failed: %s", exc)
            return

        self.running = True

        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Frame capture failed – skipping.")
                    continue

                # ── Face detection & privacy processing ──────
                is_unauth = (self.mode == self.MODE_UNAUTHORIZED)
                frame, face_count = self.detector.process_frame(frame, is_unauth)

                # ── Alerts (only in unauthorized mode) ───────
                if is_unauth and face_count > 0:
                    self.alerter.trigger_alert(frame, face_count)

                # ── HUD overlay ──────────────────────────────
                self._update_fps()
                self._draw_hud(frame, face_count)

                # ── Display ───────────────────────────────────
                cv2.imshow("BlurSense AI – Privacy CCTV", frame)

                # ── Keyboard (wait 1 ms between frames) ───────
                key = cv2.waitKey(1)
                if not self._handle_key(key):
                    break

        except KeyboardInterrupt:
            logger.info("Interrupted via keyboard (Ctrl+C).")
        except Exception as exc:
            logger.exception("Unexpected error in main loop: %s", exc)
        finally:
            self._release_camera()
            logger.info("BlurSense AI shut down cleanly.")


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = BlurSenseApp()
    app.run()
