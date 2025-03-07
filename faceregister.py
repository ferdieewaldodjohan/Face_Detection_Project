import dlib
import cv2
import os
import numpy as np
import time

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")

save_directory = "known_faces"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

name = input("Masukkan nama Anda: ")
user_folder = os.path.join(save_directory, name)

if os.path.exists(user_folder):
    existing_photos = [f for f in os.listdir(user_folder) if f.endswith(".jpg")]
    if len(existing_photos) >= 9:
        for file in existing_photos:
            os.remove(os.path.join(user_folder, file))
        # print("Foto lama telah dihapus. Mengambil foto baru...")
else:
    os.makedirs(user_folder)

face_positions = [
    "Menghadap Depan",
    "Menghadap Kanan",
    "Menghadap Kiri",
    "Menghadap Atas",
    "Menghadap Kanan Atas",
    "Menghadap Kiri Atas",
    "Menghadap Bawah",
    "Menghadap Kanan Bawah",
    "Menghadap Kiri Bawah"
]

def get_face_direction(landmarks):
    nose = landmarks.part(30)  
    left_eye = landmarks.part(36)  
    right_eye = landmarks.part(45)  

    eye_diff = abs(left_eye.x - right_eye.x)
    nose_to_eye_left = abs(nose.x - left_eye.x)
    nose_to_eye_right = abs(nose.x - right_eye.x)

    if eye_diff > 20 and abs(nose_to_eye_left - nose_to_eye_right) < 15:
        return "Menghadap Depan"
    elif nose_to_eye_left > nose_to_eye_right + 10:
        return "Menghadap Kanan"
    elif nose_to_eye_right > nose_to_eye_left + 10:
        return "Menghadap Kiri"
    return "Tidak Terdeteksi"

video_capture = cv2.VideoCapture(0)

captured_photos = 0
face_locked_time = 0
LOCK_DELAY = 2  

# print("\nSilakan hadapkan wajah sesuai instruksi di layar.")
# print("Tutup jendela kamera untuk keluar kapan saja.")

while captured_photos < 9:
    ret, frame = video_capture.read()
    if not ret:
        # print("Gagal mengambil frame.")
        break

    frame = cv2.flip(frame, 1)  

    faces = face_detector(frame)

    if len(faces) == 1:
        face = faces[0]
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        margin = 30  
        x = max(x - margin, 0)
        y = max(y - margin, 0)
        w = min(w + 2 * margin, frame.shape[1] - x)
        h = min(h + 2 * margin, frame.shape[0] - y)

        landmarks = shape_predictor(frame, face)
        current_position = get_face_direction(landmarks)
        target_position = face_positions[captured_photos]

        if target_position in ["Menghadap Atas", "Menghadap Bawah"]:
            target_position = "Menghadap Depan"
        elif target_position in ["Menghadap Kanan Atas", "Menghadap Kanan Bawah", "Menghadap Kanan"]:
            target_position = "Menghadap Kanan"
        elif target_position in ["Menghadap Kiri Atas", "Menghadap Kiri Bawah", "Menghadap Kiri"]:
            target_position = "Menghadap Kiri"

        cv2.putText(frame, f"Arahkan wajah: {face_positions[captured_photos]}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Deteksi: {current_position}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if current_position == target_position:
            if face_locked_time == 0:
                face_locked_time = time.time()
            elif time.time() - face_locked_time >= LOCK_DELAY:
                face_img = frame[y:y + h, x:x + w]
                filename = os.path.join(user_folder, f"{name}_{face_positions[captured_photos]}.jpg")
                cv2.imwrite(filename, face_img)
                captured_photos += 1
                # print(f"Foto {captured_photos}/9 disimpan: {filename}")
                face_locked_time = 0
        else:
            face_locked_time = 0

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 🟢 Bounding Box Lebih Besar

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break


video_capture.release()
cv2.destroyAllWindows()
# print("Registrasi wajah selesai.")
