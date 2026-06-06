import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import threading
from deepface import DeepFace
import urllib.request

# --- KONFIGURASI ---
DB_PATH = "dataset"
SCAN_INTERVAL = 30  # AI scan setiap 30 frame (~1 detik)

# 1. OTOMATISASI DOWNLOAD MODEL BLAZEFACE
MODEL_PATH = "blaze_face_short_range.tflite"
if not os.path.exists(MODEL_PATH):
    print("[INFO] Mengunduh model MediaPipe face detection...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        MODEL_PATH
    )
    print("[INFO] Model berhasil diunduh.")

# 2. INISIALISASI MEDIAPIPE TASKS API
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.6
)
face_detector = mp_vision.FaceDetector.create_from_options(options)

# 3. INISIALISASI KAMERA
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW untuk Windows

# --- SET EXPOSURE MANUAL (Anti-Silau) ---
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = matikan auto-exposure
cap.set(cv2.CAP_PROP_EXPOSURE, -4)         # Tuning: -4 (terang) s/d -8 (gelap)

# Verifikasi apakah exposure berhasil di-set
actual_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
print(f"[INFO] Exposure kamera di-set ke: {actual_exposure}")

# --- STRUKTUR DATA TRACKING (Anti-Amnesia / Anti-Jitter) ---
face_tracker = {}  # Format: {face_id: (center_x, center_y)}
face_names = {}    # Format: {face_id: "NAMA"}
next_face_id = 0
scan_lock = threading.Lock()
frame_count = 0

print("\n=== SISTEM ABSENSI MULTI-FACE AI (ANTI-SILAU) READY ===")
print("Tekan [Q] untuk Keluar\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape
    current_frame_faces_to_scan = []

    # 4. DETEKSI WAJAH (Setiap Frame)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    detection_result = face_detector.detect(mp_image)

    new_tracker = {}

    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box

            # FITUR: BBox Padding 15% agar dahi/rahang ikut terpotong untuk DeepFace
            pad_w = int(bbox.width * 0.15)
            pad_h = int(bbox.height * 0.15)

            xmin = max(0, bbox.origin_x - pad_w)
            ymin = max(0, bbox.origin_y - pad_h)
            width = min(w - xmin, bbox.width + (pad_w * 2))
            height = min(h - ymin, bbox.height + (pad_h * 2))

            # Hitung titik tengah wajah (Centroid) untuk pelacakan
            cx = xmin + width // 2
            cy = ymin + height // 2

            # FITUR: Centroid Tracking (Mencari ID terdekat dari frame sebelumnya)
            matched_id = None
            min_dist = 70  # Batas toleransi pergeseran wajah (dalam piksel)

            for fid, (tx, ty) in face_tracker.items():
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    matched_id = fid

            # Jika wajah benar-benar baru, daftarkan ID baru
            if matched_id is None:
                matched_id = next_face_id
                next_face_id += 1
                with scan_lock:
                    face_names[matched_id] = "Scanning..."

            # Simpan posisi wajah saat ini untuk referensi frame berikutnya
            new_tracker[matched_id] = (cx, cy)

            # Ambil status nama dari memori global
            with scan_lock:
                name = face_names.get(matched_id, "Scanning...")

            # Set warna visual berdasarkan status nama
            if name == "Scanning...":
                color = (0, 165, 255)   # Oranye (Proses)
            elif name == "Unknown":
                color = (0, 0, 255)     # Merah (Tidak Dikenal)
            else:
                color = (0, 255, 0)     # Hijau (Terverifikasi)

            # Gambar kotak dan teks nama di layar
            cv2.rectangle(frame, (xmin, ymin), (xmin + width, ymin + height), color, 2)
            cv2.putText(frame, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Masukkan data ke antrean untuk discan DeepFace saat interval tercapai
            current_frame_faces_to_scan.append((matched_id, xmin, ymin, width, height))

    # Perbarui tracker utama
    face_tracker = new_tracker

    # 5. REKOGNISI WAJAH (Hanya saat Interval & Menggunakan Background Thread)
    if frame_count % SCAN_INTERVAL == 0 and current_frame_faces_to_scan:

        def do_face_recognition(frame_to_scan, faces_to_scan):
            global face_names
            temp_results = {}

            for fid, x, y, wd, ht in faces_to_scan:
                if wd <= 0 or ht <= 0:
                    continue

                # Potong gambar wajah
                face_img = frame_to_scan[y:y + ht, x:x + wd]
                if face_img.size == 0:
                    continue

                # NORMALISASI CAHAYA & KONTRAS (CLAHE) ANTI-SILAU
                try:
                    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
                    channels = list(cv2.split(ycrcb))

                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    channels[0] = clahe.apply(channels[0])

                    ycrcb = cv2.merge(channels)
                    face_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                except Exception as e:
                    print(f"[WARN] Gagal memproses penyeimbang cahaya: {e}")

                try:
                    # Eksekusi pencocokan ke DeepFace dataset
                    result = DeepFace.find(
                        img_path=face_img,
                        db_path=DB_PATH,
                        detector_backend='skip',
                        enforce_detection=False,
                        silent=True
                    )

                    if len(result) > 0 and not result[0].empty:
                        matched_path = result[0]['identity'][0]
                        file_name = os.path.basename(matched_path)
                        nama_karyawan = os.path.splitext(file_name)[0].upper()
                        temp_results[fid] = nama_karyawan
                    else:
                        temp_results[fid] = "Unknown"

                except Exception as e:
                    temp_results[fid] = "Unknown"

            # Sinkronisasi hasil ke memori global
            with scan_lock:
                for fid, name in temp_results.items():
                    face_names[fid] = name

        # Ambil snapshot frame lalu lempar ke background thread
        frame_copy = frame.copy()
        threading.Thread(
            target=do_face_recognition,
            args=(frame_copy, current_frame_faces_to_scan),
            daemon=True
        ).start()

    # Tampilkan output video
    cv2.imshow("Multi-Face Absensi (Optimized)", frame)

    # Keluar dengan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. CLEANUP RESOURCES
cap.release()
face_detector.close()
cv2.destroyAllWindows()