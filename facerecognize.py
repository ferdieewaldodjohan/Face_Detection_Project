import cv2
import dlib
import pickle
import os
import numpy as np
from tqdm import tqdm

KNOWN_FACES_DIR = ["known_faces", "augmented_faces"]
ENCODING_FILE = "face_encodings_dlib.pkl"

if os.path.exists(ENCODING_FILE):
    os.remove(ENCODING_FILE)

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")
face_recognizer = dlib.face_recognition_model_v1("taguchi_face_recognition_resnet_model_v1.dat")

encodings = []
names = []

for dir in KNOWN_FACES_DIR:
    if not os.path.exists(dir):
        # print(f"❌ Folder {dir} tidak ditemukan, dilewati.")
        continue

    for person_name in tqdm(os.listdir(dir), desc=f"Processing {dir}"):
        folder_path = os.path.join(dir, person_name)

        if not os.path.isdir(folder_path):
            continue  

        for filename in tqdm(os.listdir(folder_path), desc=f"Processing {person_name}", leave=False):
            image_path = os.path.join(folder_path, filename)

            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue  

            image = cv2.imread(image_path)
            if image is None:
                # print(f"❌ Gagal membaca gambar: {image_path}")
                continue

            image = cv2.resize(image, (600, 600))
            faces = face_detector(image)

            if len(faces) == 0:
                # print(f"⚠️ Wajah tidak ditemukan di {image_path}, tetap dimasukkan ke dalam.")
                encoding = np.zeros(128)  # Encoding default untuk gambar tanpa wajah
                assigned_name = person_name  # Label gambar tanpa wajah sebagai "ferdie"
            else:
                landmarks = shape_predictor(image, faces[0])
                encoding = np.array(face_recognizer.compute_face_descriptor(image, landmarks))
                assigned_name = person_name  # Gunakan nama folder asli jika wajah terdeteksi

            encodings.append(encoding)
            names.append(assigned_name)

            # print(f"✅ Gambar {filename} berhasil dimasukkan sebagai '{assigned_name}'.")

# **Simpan semua file ke dalam .pkl**
with open(ENCODING_FILE, "wb") as f:
    pickle.dump((encodings, names), f)

# print(f"\n💾 Berhasil menyimpan {len(names)} gambar ke dalam {ENCODING_FILE}")
