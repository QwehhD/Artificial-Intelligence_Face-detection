import cv2
import numpy as np
import face_recognition
import os

# ============================================================
# INTERFACE 1: LOADING & ENCODING DATASET WAJAH
# ============================================================
DATASET_PATH = 'dataset'
known_face_encodings = []
known_face_names = []

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
    print(f"[AI INFO] Folder '{DATASET_PATH}' dibuat. Silakan isi foto wajah dulu!")

print("[AI INFO] Membaca dataset dan melakukan ekstraksi vektor wajah...")
for file_name in os.listdir(DATASET_PATH):
    img_path = f"{DATASET_PATH}/{file_name}"
    image = cv2.imread(img_path)
    
    if image is None:
        continue
        
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        name_id = os.path.splitext(file_name)[0].upper()
        known_face_names.append(name_id)
        print(f" -> Berhasil encoding wajah: {name_id}")
    else:
        print(f" -> [WARN] Wajah tidak terdeteksi pada file: {file_name}")

print(f"[AI INFO] Total wajah terdaftar secara lokal: {len(known_face_names)}")

# ============================================================
# INTERFACE 2: REAL-TIME CAPTURE & MATCHING LOOP
# ============================================================
cap = cv2.VideoCapture(0)

# Inisialisasi variabel untuk optimasi Frame Skipping
process_this_frame = True
face_locations = []
face_encodings = []
face_names = []
face_colors = []

while True:
    success, frame = cap.read()
    if not success:
        break

    # Jalankan proses berat AI hanya jika flag bernilai True (bergantian setiap frame)
    if process_this_frame:
        # Perkecil frame ke 0.25x agar proses encoding berjalan lebih ringan
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 1. Deteksi lokasi dan bentuk vektor wajah
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_colors = []

        # 2. Proses Klasifikasi / Pencocokan Wajah
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            name = "UNKNOWN"
            color = (0, 0, 255) # Merah untuk asing

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    color = (0, 255, 0) # Hijau untuk terdaftar
                    
                    similarity_percentage = (1 - face_distances[best_match_index]) * 100
                    print(f"[AI MATCH] Teridentifikasi: {name} | Jarak: {face_distances[best_match_index]:.4f} | Akurasi: {similarity_percentage:.2f}%")
            
            face_names.append(name)
            face_colors.append(color)

    # Balik nilai flag (Frame berikutnya dilewati dari komputasi AI berat)
    process_this_frame = not process_this_frame

    # Render Bounding Box di Layar Monitor (Menggunakan data dari cache frame sebelumnya jika sedang skip)
    for face_location, name, color in zip(face_locations, face_names, face_colors):
        # Kembalikan koordinat kotak wajah ke ukuran semula (dikali 4)
        top, right, bottom, left = face_location
        top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

        # Render grafik kotak dan text ke frame asli
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('AI Face Recognition Server (Lokal)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()