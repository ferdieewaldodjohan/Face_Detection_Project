import cv2
import dlib
import faiss
import json
import numpy as np
import datetime
import firebase_admin
from firebase_admin import credentials, db
import pytz

# Firebase Setup
cred = credentials.Certificate("face-detection-9f00c-firebase-adminsdk-fbsvc-cea00cf052.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-detection-9f00c-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

log_ref = db.reference("face_logs")  # Firebase path for logs

# Path model
shape_predictor_path = "shape_predictor_68_face_landmarks_GTX.dat"
face_recognizer_path = "taguchi_face_recognition_resnet_model_v1.dat"

# FAISS Index & Names
FAISS_INDEX_FILE = "face_encodings.index"
NAMES_FILE = "face_data.json"  # Updated to match new format
VECTOR_DIM = 128  # Face encoding dimension

# Load Models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)

faiss_index = faiss.read_index(FAISS_INDEX_FILE)
with open(NAMES_FILE, "r") as f:
    known_data = json.load(f)  # Updated to match new format

accuracy_threshold = 60.0  # Adjust threshold

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
        print("Logged to Firebase:", log_entry)

def main_loop():
    video_capture = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                continue

            faces = face_detector(frame)
            detected_name = "Unknown"
            detected_nrp = "Unknown"

            if len(faces) > 0:
                face = max(faces, key=lambda f: f.width() * f.height())

                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                landmarks = shape_predictor(frame, face)
                encoding = np.array(face_recognizer.compute_face_descriptor(frame, landmarks)).astype(np.float32).reshape(1, -1)

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

                    # Push to Firebase only if it's not "Unknown"
                    push_to_firebase(detected_name, detected_nrp)

                # Draw on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{detected_name} ({detected_nrp}) ({accuracy:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("x"):
                break

    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
