import cv2
import dlib
import faiss
import os
import json
import numpy as np
from tqdm import tqdm

# Path model
shape_predictor_path = "shape_predictor_68_face_landmarks_GTX.dat"
face_recognizer_path = "taguchi_face_recognition_resnet_model_v1.dat"

# Direktori wajah
KNOWN_FACES_DIR = ["known_faces", "augmented_faces"]
FAISS_INDEX_FILE = "face_encodings.index"
NAMES_FILE = "face_data.json"  # Simpan sebagai dict {"Nama_NRP": index}
VECTOR_DIM = 128  # Dimensi encoding wajah

# Load Dlib models
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(shape_predictor_path)
face_recognizer = dlib.face_recognition_model_v1(face_recognizer_path)

# Inisialisasi FAISS Index
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
name_nrp_list = []

# Loop semua folder wajah
for dir in KNOWN_FACES_DIR:
    if not os.path.exists(dir):
        continue

    for person_folder in tqdm(os.listdir(dir), desc=f"Processing {dir}"):
        folder_path = os.path.join(dir, person_folder)
        if not os.path.isdir(folder_path):
            continue

        # Parse Nama dan NRP dari folder
        if "_" in person_folder:
            name, nrp = person_folder.rsplit("_", 1)
        else:
            name, nrp = person_folder, "Unknown"

        for filename in tqdm(os.listdir(folder_path), desc=f"Processing {person_folder}", leave=False):
            image_path = os.path.join(folder_path, filename)
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue  

            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.resize(image, (600, 600))
            faces = face_detector(image)

            if len(faces) == 0:
                encoding = np.zeros(VECTOR_DIM)  # Jika wajah tidak ditemukan
            else:
                landmarks = shape_predictor(image, faces[0])
                encoding = np.array(face_recognizer.compute_face_descriptor(image, landmarks)).astype(np.float32)

            # Tambahkan encoding ke FAISS
            faiss_index.add(np.array([encoding], dtype=np.float32))
            name_nrp_list.append({"name": name, "nrp": nrp})

# Simpan FAISS Index dan data nama + NRP
faiss.write_index(faiss_index, FAISS_INDEX_FILE)

with open(NAMES_FILE, "w") as f:
    json.dump(name_nrp_list, f, indent=4)

print(f"\n✅ Berhasil menyimpan {len(name_nrp_list)} wajah ke FAISS: {FAISS_INDEX_FILE} & {NAMES_FILE}")