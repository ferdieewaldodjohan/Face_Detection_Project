import cv2
import dlib
import pickle
import numpy as np
from scipy.spatial import distance as dist

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")
face_recognizer = dlib.face_recognition_model_v1("taguchi_face_recognition_resnet_model_v1.dat") 

ENCODING_FILE = "face_encodings_dlib.pkl"

try:
    with open(ENCODING_FILE, "rb") as f:
        known_encodings, known_names = pickle.load(f)
    # print(f"Loaded {len(known_names)} known faces.")
except FileNotFoundError:
    # print("File encoding tidak ditemukan! Jalankan facerecognize.py dulu.")
    exit()

accuracy_threshold = 40.0 
EAR_THRESHOLD = 0.25
BLINK_FRAMES = 2

blink_counter = 0
blink_verified = False  

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def main_loop():
    global blink_counter, blink_verified
    video_capture = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                # print("Failed to capture frame.")
                continue

            faces = face_detector(frame)

            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                landmarks = shape_predictor(frame, face)
                
                left_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
                right_eye_pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
                
                left_EAR = eye_aspect_ratio(left_eye_pts)
                right_EAR = eye_aspect_ratio(right_eye_pts)
                avg_EAR = (left_EAR + right_EAR) / 2.0

                if avg_EAR < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= BLINK_FRAMES:
                        blink_verified = True
                        # print("✅ Kedipan terdeteksi! Memulai pengenalan wajah...")
                    blink_counter = 0
                
                if not blink_verified:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Silakan berkedip!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    continue  
                
                encoding = np.array(face_recognizer.compute_face_descriptor(frame, landmarks))

                distances = [np.linalg.norm(encoding - known_encoding) for known_encoding in known_encodings]
                min_distance = min(distances)
                match_index = distances.index(min_distance)
                
                if min_distance < 0.6:
                    name = known_names[match_index]
                    accuracy = (1 - min_distance / 0.6) * 100
                else:
                    name = "Unknown"
                    accuracy = 0

                if accuracy < accuracy_threshold:
                    name = "Unknown"
                    color = (0, 0, 255)  
                else:
                    color = (0, 255, 0)  
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{name} ({accuracy:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            cv2.imshow("Face Recognition", frame)

            if len(faces) == 0:
                blink_verified = False  

            if cv2.waitKey(1) & 0xFF == ord("x") or cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
