import dlib
import cv2
import os
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, db
import tqdm
import random
import faiss
import json

# Initialize Firebase
cred = credentials.Certificate("face-detection-9f00c-firebase-adminsdk-fbsvc-cea00cf052.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://face-detection-9f00c-default-rtdb.asia-southeast1.firebasedatabase.app/"
})
users_ref = db.reference("registered_users")  # Firebase path for users list

# Load Dlib Models (shared across all functionalities)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks_GTX.dat")
face_recognizer = dlib.face_recognition_model_v1("taguchi_face_recognition_resnet_model_v1.dat")

# Directory setup
save_directory = "known_faces"
augmented_directory = "augmented_faces"
FAISS_INDEX_FILE = "face_encodings.index"
NAMES_FILE = "face_data.json"
VECTOR_DIM = 128  # Dimensi encoding wajah

if not os.path.exists(save_directory):
    os.makedirs(save_directory)
if not os.path.exists(augmented_directory):
    os.makedirs(augmented_directory)

# === STEP : FACEREGISTER FUNCTIONS === 
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

def register_face():
    # Get User Input
    name = input("Masukkan nama Anda: ")
    nrp = input("Masukkan NRP Anda: ")

    # Check if the user is already registered in Firebase
    existing_users = users_ref.get() or {}  # Get existing users or empty dict

    # Create user folder if new
    user_folder = os.path.join(save_directory, f"{name}_{nrp}")
    if os.path.exists(user_folder):
        existing_photos = [f for f in os.listdir(user_folder) if f.endswith(".jpg")]
        if len(existing_photos) >= 9:
            for file in existing_photos:
                os.remove(os.path.join(user_folder, file))
    else:
        os.makedirs(user_folder)

    video_capture = cv2.VideoCapture(0)
    captured_photos = 0
    face_locked_time = 0
    LOCK_DELAY = 2  

    print("⚠️ Silakan hadapkan wajah sesuai instruksi di layar.")
    print("❌ Tekan 'X' untuk keluar kapan saja.")

    while captured_photos < 9:
        ret, frame = video_capture.read()
        if not ret:
            print("❌ Gagal mengambil frame.")
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
                    filename = os.path.join(user_folder, f"{name}_{nrp}_{face_positions[captured_photos]}.jpg")
                    cv2.imwrite(filename, face_img)
                    captured_photos += 1
                    print(f"📸 Foto {captured_photos}/9 disimpan: {filename}")
                    face_locked_time = 0
            else:
                face_locked_time = 0

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord("x"):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if not any(user["nrp"] == nrp for user in existing_users.values()):
        # Store user in Firebase
        new_user_ref = users_ref.push()
        new_user_ref.set({
            "name": name,
            "nrp": nrp
        })
        print("✅ Registrasi wajah selesai & data tersimpan di Firebase.")
    
    return user_folder

# === STEP : AUGMENTASI FUNCTIONS === 
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def scale_image(image, scale_factor):
    scaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled

def translate_image(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    translated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return translated

def flip_image(image, flip_code):
    flipped = cv2.flip(image, flip_code)
    return flipped

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 400
        else:
            shadow = 0
            highlight = 400 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 200 * (contrast + 150) / (150 * (200 - contrast))
        alpha_c = f
        gamma_c = 150 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def add_gaussian_noise(image, mean=0, sigma=20, alpha=0.5):
    gauss = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    return noisy

def crop_image(image, x, y, width, height):
    cropped = image[y:y+height, x:x+width]
    return cropped

def perspective_transform(image, src_points, dst_points):
    h, w = image.shape[:2]
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    transformed = cv2.warpPerspective(image, M, (w, h))
    return transformed

def color_jitter(image, hue_delta=18, saturation_scale=1.5, brightness_scale=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(int) + random.randint(-hue_delta, hue_delta)
    hue = np.clip(hue, 0, 179).astype(np.uint8)
    hsv[:, :, 0] = hue
    saturation = np.clip(hsv[:, :, 1] * random.uniform(1/saturation_scale, saturation_scale), 0, 255).astype(np.uint8)
    hsv[:, :, 1] = saturation
    brightness = np.clip(hsv[:, :, 2] * random.uniform(1/brightness_scale, brightness_scale), 0, 255).astype(np.uint8)
    hsv[:, :, 2] = brightness
    jittered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return jittered

def convert_to_grayscale(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale

def generate_specific_occlusions(face_frame, adjusted_landmarks):
    occlusions = []
    features = [
        {'name': 'left_eye', 'indices': list(range(36, 42))},
        {'name': 'right_eye', 'indices': list(range(42, 48))},
        {'name': 'nose', 'indices': list(range(30, 36))},
        {'name': 'mouth', 'indices': list(range(48, 68))},
        {'name': 'chin', 'indices': list(range(6, 12))}
    ]
    
    for feature in features:
        indices = feature['indices']
        points = [adjusted_landmarks[i] for i in indices if i < len(adjusted_landmarks)]
        if not points:
            continue
        
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min = int(min(x_coords))
        x_max = int(max(x_coords))
        y_min = int(min(y_coords))
        y_max = int(max(y_coords))
        
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            continue
        
        expand_x = int(0.2 * width)
        expand_y = int(0.2 * height)
        
        x_min_new = max(0, x_min - expand_x)
        x_max_new = min(face_frame.shape[1], x_max + expand_x)
        y_min_new = max(0, y_min - expand_y)
        y_max_new = min(face_frame.shape[0], y_max + expand_y)
        
        if x_max_new <= x_min_new or y_max_new <= y_min_new:
            continue
        
        occluded = face_frame.copy()
        occluded[y_min_new:y_max_new, x_min_new:x_max_new] = 0
        occlusions.append(occluded)
    
    return occlusions

def add_motion_blur(image, kernel_size=5):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)

def random_cutout(image, num_holes=5, min_hole_size=5, max_hole_size=10):
    h, w = image.shape[:2]
    image_copy = image.copy()  
    for _ in range(num_holes):
        hole_h = random.randint(min_hole_size, max_hole_size)
        hole_w = random.randint(min_hole_size, max_hole_size)
        x = random.randint(0, max(0, w - hole_w - 1))
        y = random.randint(0, max(0, h - hole_h - 1))
        if len(image.shape) == 3:
            image_copy[y:y + hole_h, x:x + hole_w] = (0, 0, 0)
        else:  
            image_copy[y:y + hole_h, x:x + hole_w] = 0
    return image_copy

def augment_image(image):
    augmented_images = []
    h, w = image.shape[:2]  

    # Scale
    augmented_images.append(scale_image(image, 2))
    augmented_images.append(scale_image(image, 1))

    # Translate
    augmented_images.append(translate_image(image, 10, 0))
    augmented_images.append(translate_image(image, -10, 0))
    augmented_images.append(translate_image(image, 0, 10))
    augmented_images.append(translate_image(image, 0, -10))

    # Brightness and Contrast
    augmented_images.append(adjust_brightness_contrast(image, 40, 40))
    augmented_images.append(adjust_brightness_contrast(image, 60, 60))
    augmented_images.append(adjust_brightness_contrast(image, -60, -60))
    augmented_images.append(adjust_brightness_contrast(image, -100, -100))

    # Gaussian Noise
    augmented_images.append(add_gaussian_noise(image, 0, 20, 0.5))

    # Perspective Transform
    src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_points[1] += (10, 0)
    dst_points[2] += (0, 10)
    augmented_images.append(perspective_transform(image, src_points, dst_points))

    # Color Jitter
    augmented_images.append(color_jitter(image))

    return augmented_images

def augment_faces(user_folder):
    output_folder_path = os.path.join(augmented_directory, os.path.basename(user_folder))
    os.makedirs(output_folder_path, exist_ok=True)
    
    for filename in tqdm(os.listdir(user_folder), desc=f"Processing Images in {os.path.basename(user_folder)}"):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(user_folder, filename)
            image = cv2.imread(image_path)
            augmented_images = []
            
            if image is None:
                continue
            
            if image.shape[0] > 180 and image.shape[1] > 180:
                faces = face_detector(image)
                
                if len(faces) == 0:
                    os.remove(image_path)
                    continue
                
                for face in faces:
                    shape = shape_predictor(image, face)
                    x_min = min([shape.part(i).x for i in range(68)])
                    x_max = max([shape.part(i).x for i in range(68)])
                    y_min = min([shape.part(i).y for i in range(68)])
                    y_max = max([shape.part(i).y for i in range(68)])
                    
                    margin_x = int((x_max - x_min) * 0.2)
                    margin_y = int((y_max - y_min) * 0.3)
                    
                    x1 = max(0, x_min - margin_x)
                    y1 = max(0, y_min - margin_y)
                    x2 = min(image.shape[1], x_max + margin_x)
                    y2 = min(image.shape[0], y_max + margin_y)
                    
                    face_frame = image[y1:y2, x1:x2]
                    
                    if face_frame.shape[0] < 50 or face_frame.shape[1] < 50:
                        continue
                    
                    adjusted_landmarks = [(shape.part(i).x - x1, shape.part(i).y - y1) for i in range(68)]
                    
                    current_augmented = augment_image(face_frame)
                    augmented_images.extend(current_augmented)
                    
                    specific_occlusions = generate_specific_occlusions(face_frame, adjusted_landmarks)
                    augmented_images.extend(specific_occlusions)
            else:
                augmented_images = augment_image(image)
            
            for i, augmented_image in enumerate(augmented_images):
                output_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg")
                cv2.imwrite(output_path, augmented_image)

# === STEP : FACERECOGNITION FUNCTIONS ===
def process_faces_for_recognition():
    KNOWN_FACES_DIR = [save_directory, augmented_directory]
    faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
    name_nrp_list = []

    for dir in KNOWN_FACES_DIR:
        if not os.path.exists(dir):
            continue

        for person_folder in tqdm(os.listdir(dir), desc=f"Processing {dir}"):
            folder_path = os.path.join(dir, person_folder)
            if not os.path.isdir(folder_path):
                continue

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
                    encoding = np.zeros(VECTOR_DIM)
                else:
                    landmarks = shape_predictor(image, faces[0])
                    encoding = np.array(face_recognizer.compute_face_descriptor(image, landmarks)).astype(np.float32)

                faiss_index.add(np.array([encoding], dtype=np.float32))
                name_nrp_list.append({"name": name, "nrp": nrp})

    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    with open(NAMES_FILE, "w") as f:
        json.dump(name_nrp_list, f, indent=4)
    
    print(f"\n✅ Berhasil menyimpan {len(name_nrp_list)} wajah ke FAISS: {FAISS_INDEX_FILE} & {NAMES_FILE}")

### MAIN EXECUTION ###
if __name__ == "__main__":
    # Step 1: Register a new user's face
    user_folder = register_face()
    
    # Step 2: Augment the captured images
    augment_faces(user_folder)
    
    # Step 3: Process all faces for recognition
    process_faces_for_recognition()