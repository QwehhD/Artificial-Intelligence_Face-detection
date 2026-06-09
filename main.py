from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import face_recognition
import os

app = FastAPI(
    title="AI Face Recognition Server",
    description="API Server Absensi Wajah untuk Next.js Integration"
)

# Konfigurasi CORS agar aplikasi Next.js (dari port manapun) bisa menembak API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Pada production, batasi ke URL Next.js kamu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# INITIALIZATION: LOADING DATASET WAJAH SECARA CACHING
# ============================================================
DATASET_PATH = 'dataset'
known_face_encodings = []
known_face_names = []

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
    print(f"[AI SERVER] Folder '{DATASET_PATH}' dibuat. Silakan masukkan template foto wajah.")

print("[AI SERVER] Membaca dataset dan mengekstrak vektor wajah ke memori...")
for file_name in os.listdir(DATASET_PATH):
    img_path = os.path.join(DATASET_PATH, file_name)
    image = cv2.imread(img_path)
    
    if image is None:
        continue
        
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    
    if len(encodings) > 0:
        known_face_encodings.append(encodings[0])
        name_id = os.path.splitext(file_name)[0].upper()
        known_face_names.append(name_id)
        print(f" -> Terdaftar: {name_id}")
    else:
        print(f" -> [WARN] Wajah gagal diekstrak pada file: {file_name}")

print(f"[AI SERVER] Inisialisasi selesai. Total database lokal: {len(known_face_names)} wajah.")

# ============================================================
# API ENDPOINT: VERIFIKASI WAJAH / ABSENSI
# ============================================================
@app.post("/api/absent")
async def absent_verification(file: UploadFile = File(...)):
    try:
        # 1. Mengubah file gambar kiriman dari Next.js menjadi format OpenCV Matrix
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "error", "message": "File gambar corrupt atau tidak valid"}

        # 2. Optimasi Skala: Perkecil resolusi ke 0.25x agar inferensi CPU cloud cepat
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 3. Deteksi lokasi wajah dan ekstraksi 128-dimensional embedding
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Jika tidak ada wajah sama sekali di dalam frame yang dikirim
        if len(face_encodings) == 0:
            return {
                "status": "success",
                "match": False,
                "name": "NO_FACE_DETECTED",
                "accuracy": 0.0
            }

        # 4. Proses Klasifikasi Jarak Euclidean (Pencocokan)
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    similarity_percentage = (1 - face_distances[best_match_index]) * 100
                    
                    print(f"[AI MATCH] Teridentifikasi: {name} (Akurasi: {similarity_percentage:.2f}%)")
                    return {
                        "status": "success",
                        "match": True,
                        "name": name,
                        "accuracy": round(similarity_percentage, 2)
                    }

        # Jika ada wajah namun tidak cocok dengan yang ada di dataset
        return {
            "status": "success",
            "match": False,
            "name": "UNKNOWN",
            "accuracy": 0.0
        }

    except Exception as e:
        return {"status": "error", "message": f"Internal Server Error: {str(e)}"}