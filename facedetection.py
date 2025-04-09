import cv2
import dlib
import faiss
import json
import numpy as np
import datetime
import firebase_admin
from firebase_admin import credentials, db
import pytz
import time
import threading
import RPi.GPIO as GPIO

# Firebase Setup
cred = credentials.Certificate("face-detection-9f00c-firebase-adminsdk-fbsvc-cea00cf052.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-detection-9f00c-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

log_ref = db.reference("face_logs")

# Path model
shape_predictor_path = "shape_predictor_68_face_landmarks_GTX.dat"
face_recognizer_path = "taguchi_face_recognition_resnet_model_v1.dat"

# FAISS Index & Names
FAISS_INDEX_FILE = "face_encodings.index"
NAMES_FILE = "face_data.json"
VECTOR_DIM = 128

# Load Models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)

faiss_index = faiss.read_index(FAISS_INDEX_FILE)
with open(NAMES_FILE, "r") as f:
    known_data = json.load(f)

accuracy_threshold = 60.0
hold_time = 3  # Seconds to maintain confidence
frame_skip = 2  # Process every 2nd frame

# Track detected faces over time
detection_timers = {}
lock = threading.Lock()

# Store the last detected person
last_detected_name = "No Person Detected"
last_detected_nrp = "-"

# === GPIO Setup ===
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)  # Solenoid
GPIO.setup(13, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Limit switch

def bukaSolenoid():
    GPIO.output(11, True)

def tutupSolenoid():
    GPIO.output(11, False)

def is_door_closed():
    return GPIO.input(13) == 0  # adjust logic as needed

def push_to_firebase(name, nrp):
    """Push detected face name, NRP & timestamp to Firebase (excluding 'Unknown')"""
    if name != "Unknown":
        local_tz = pytz.timezone("Asia/Jakarta")
        timestamp = datetime.datetime.now(local_tz).isoformat()
        log_entry = {
            "timestamp": timestamp,
            "detected_name": name,
            "nrp": nrp
        }
        log_ref.push(log_entry)
        print(f"✅ Logged to Firebase: {log_entry}")


def track_face(name, nrp, accuracy):
    """Tracks face detection over time and logs if confidence is held for 5 sec"""
    global last_detected_name, last_detected_nrp
    current_time = time.time()

    with lock:
        if accuracy >= accuracy_threshold:
            last_detected_name = name
            last_detected_nrp = nrp

            if name not in detection_timers:
                detection_timers[name] = current_time  # Start timer

            elapsed_time = current_time - detection_timers[name]

            if elapsed_time >= hold_time:
                push_to_firebase(name, nrp)
                print(f"[✅] Recognized {name} - Opening Door")
                bukaSolenoid()

                # Wait until door is closed
                timeout = time.time() + 10
                while not is_door_closed() and time.time() < timeout:
                    time.sleep(0.2)

                print("[🔒] Closing Door")
                tutupSolenoid()
                del detection_timers[name]  # Reset timer after logging
        else:
            if name in detection_timers:
                del detection_timers[name]  # Reset if confidence drops


def process_frame(frame):
    """Runs face detection & recognition in a separate thread"""
    global frame_skip

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    faces = face_detector(small_frame)
    detected_name = "Unknown"
    detected_nrp = "Unknown"

    if len(faces) > 0:
        face = max(faces, key=lambda f: f.width() * f.height())

        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        landmarks = shape_predictor(small_frame, face)
        encoding = np.array(face_recognizer.compute_face_descriptor(small_frame, landmarks)).astype(np.float32).reshape(1, -1)

        distances, indices = faiss_index.search(encoding, 1)
        min_distance = distances[0][0]
        match_index = indices[0][0]

        if min_distance < 0.5:
            detected_name = known_data[match_index]["name"]
            detected_nrp = known_data[match_index]["nrp"]
            accuracy = (1 - min_distance / 0.5) * 100
        else:
            detected_name = "Unknown"
            detected_nrp = "Unknown"
            accuracy = 0

        if accuracy < accuracy_threshold:
            detected_name = "Unknown"
            detected_nrp = "Unknown"
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
            track_face(detected_name, detected_nrp, accuracy)

        # Draw on original frame (scale coordinates back)
        x, y, w, h = x * 2, y * 2, w * 2, h * 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{detected_nrp} ({accuracy:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main_loop():
    video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for better performance on Windows
    frame_count = 0

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue
            if GPIO.input(15) == 0:
                print("[🔒] Opening Door")
                bukaSolenoid()
                while not is_door_closed():
                    time.sleep(0.2)

                print("[🔒] Closing Door")
                tutupSolenoid()
            frame_count += 1
            if frame_count % frame_skip == 0:  # Process only every N-th frame
                threading.Thread(target=process_frame, args=(frame.copy(),)).start()

            # Display last detected person info
            cv2.putText(frame, f"Detected: {last_detected_name} ({last_detected_nrp})",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("x"):
                break

    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print("\nExiting and cleaning up GPIO...")
        GPIO.cleanup()


if __name__ == "__main__":
    main_loop()
