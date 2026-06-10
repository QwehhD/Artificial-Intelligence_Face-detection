from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import face_recognition
import os
import requests
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="AI Face Recognition Server",
    description="API Server Absensi Wajah dengan Integrasi Supabase & Next.js"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION: SUPABASE CONFIG
# ============================================================
# Mengambil kredensial dari Environment Variables (Lokal / Railway env)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://xyz.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "your-anon-key-here")
BUCKET_NAME = "faces"  # Nama bucket storage di Supabase kamu

# Inisialisasi Klien Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Array cache di memori server
known_face_encodings = []
known_face_names = []

# ============================================================
# INITIALIZATION: LOADING DATASET DARI SUPABASE STORAGE
# ============================================================
print("[AI SERVER] Menghubungkan ke Supabase Storage...")
try:
    # 1. List semua file yang ada di dalam bucket storage 'faces'
    response = supabase.storage.from_(BUCKET_NAME).list()
    
    print("[AI SERVER] Mengunduh dataset wajah dan mengekstrak vektor...")
    for file_info in response:
        file_name = file_info['name']
        
        # Lewati file placeholder bawaan supabase jika ada
        if file_name == '.emptyFolderPlaceholder' or file_name.startswith('.'):
            continue
            
        # 2. Dapatkan Public URL untuk file tersebut
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
        
        # 3. Unduh gambar langsung ke memori RAM (Tanpa disimpan ke SSD)
        img_resp = requests.get(public_url)
        if img_resp.status_code != 200:
            print(f" -> [WARN] Gagal mengunduh file: {file_name}")
            continue
            
        # Convert bytes gambar menjadi format OpenCV Matrix
        nparr = np.frombuffer(img_resp.content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            continue
            
        # 4. Ekstrak vektor wajah menggunakan face_recognition
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        
        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            # Mengambil nama file tanpa ekstensi sebagai ID Nama (Contoh: DIKA.jpg -> DIKA)
            name_id = os.path.splitext(file_name)[0].upper()
            known_face_names.append(name_id)
            print(f" -> Terdaftar dari Cloud: {name_id}")
        else:
            print(f" -> [WARN] Wajah gagal diekstrak pada file cloud: {file_name}")

    print(f"[AI SERVER] Inisialisasi selesai. Total database cloud: {len(known_face_names)} wajah.")

except Exception as e:
    print(f"[AI SERVER ERROR] Gagal memuat database dari Supabase: {str(e)}")


# ============================================================
# API ENDPOINT: VERIFIKASI WAJAH / ABSENSI
# ============================================================
@app.post("/api/absent")
async def absent_verification(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"status": "error", "message": "File gambar corrupt atau tidak valid"}

        # Optimasi Skala ke 0.25x
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        if len(face_encodings) == 0:
            return {
                "status": "success",
                "match": False,
                "name": "NO_FACE_DETECTED",
                "accuracy": 0.0
            }

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

        return {
            "status": "success",
            "match": False,
            "name": "UNKNOWN",
            "accuracy": 0.0
        }

    except Exception as e:
        return {"status": "error", "message": f"Internal Server Error: {str(e)}"}