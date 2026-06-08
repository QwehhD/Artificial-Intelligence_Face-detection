import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import threading
from deepface import DeepFace
import urllib.request
import numpy as np

# ============================================================
# KONFIGURASI TARGET AKURASI LOKAL
# ============================================================
DB_PATH               = "dataset"
SCAN_INTERVAL         = 15  # Dipercepat agar scanning lebih responsif

# Threshold dilonggarkan maksimal agar toleran di segala kondisi kamar
CONFIDENCE_THRESHOLD  = 0.20   # Diturunkan agar mudah menerima kecocokan
ARCFACE_DIST_THRESHOLD = 0.75   # Dinaikkan (Lebih longgar untuk Cosine Distance)

TRACKER_TOLERANCE_PX  = 80
BBOX_PAD_RATIO         = 0.15
ARCFACE_INPUT_SIZE     = (112, 112)

# ============================================================
# DOWNLOAD MODEL MEDIAPIPE
# ============================================================
MODEL_CANDIDATES = [
    (
        "face_detection_full_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_full_range/float16/latest/"
        "face_detection_full_range.tflite",
        "full range"
    ),
    (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/latest/"
        "blaze_face_short_range.tflite",
        "short range (fallback)"
    ),
]

MODEL_PATH = None
for _path, _url, _label in MODEL_CANDIDATES:
    if os.path.exists(_path):
        MODEL_PATH = _path
        print(f"[INFO] Menggunakan model lokal: {_path}")
        break
    print(f"[INFO] Mengunduh model MediaPipe ({_label})...")
    try:
        urllib.request.urlretrieve(_url, _path)
        MODEL_PATH = _path
        print(f"[INFO] Model berhasil diunduh: {_path}")
        break
    except Exception as e:
        print(f"[WARN] Gagal mengunduh {_label}: {e}")

if MODEL_PATH is None:
    raise RuntimeError("[ERROR] Semua model gagal diunduh.")

# ============================================================
# INISIALISASI MEDIAPIPE
# ============================================================
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.40
)
face_detector = mp_vision.FaceDetector.create_from_options(options)

# ============================================================
# INISIALISASI KAMERA (AUTO-EXPOSURE AKTIF/DEFAULT)
# ============================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


def clean_adaptive_brightness(frame: np.ndarray) -> np.ndarray:
    """
    Pencerah gambar ringan dan natural jika ruangan mendadak agak redup,
    tanpa merusak atau memecah pixel gambar asli.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 100:
        gamma = 1.3
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(frame, table)
    return frame


def extract_name_from_path(identity_path: str) -> str:
    parts = identity_path.replace("\\", "/").split("/")
    if len(parts) >= 3:
        folder_name = parts[-2]
        if folder_name.lower() != "dataset":
            return folder_name.upper()
    filename = os.path.splitext(parts[-1])[0]
    if "_" in filename:
        filename = filename.split("_")[0]
    return filename.upper()


def distance_to_confidence(distance: float, threshold: float = ARCFACE_DIST_THRESHOLD) -> float:
    if distance <= 0.0:
        return 1.0
    if distance >= threshold:
        return 0.0
    confidence = 1.0 - (distance / threshold)
    return round(float(np.clip(confidence, 0.0, 1.0)), 4)


# ============================================================
# STRUKTUR DATA TRACKING & MAIN VARIABLES
# ============================================================
face_tracker  = {}
face_data     = {}
next_face_id  = 0
scan_lock     = threading.Lock()
frame_count   = 0

print("\n=== SYSTEM ONLINE — SETTING CAMERA DIRECTSHOW AUTO ===")
print(f"[CONFIG] Path Dataset : {os.path.abspath(DB_PATH)}\n")

# Main Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = clean_adaptive_brightness(frame)
    frame_count += 1
    h, w, _ = frame.shape
    faces_this_frame = []

    # Deteksi Wajah dengan MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result    = face_detector.detect(mp_image)
    new_tracker = {}

    if result.detections:
        for detection in result.detections:
            bbox      = detection.bounding_box
            det_score = detection.categories[0].score if detection.categories else 0.0

            if det_score < 0.45:
                continue

            pad_w = int(bbox.width  * BBOX_PAD_RATIO)
            pad_h = int(bbox.height * BBOX_PAD_RATIO)
            xmin  = max(0, bbox.origin_x - pad_w)
            ymin  = max(0, bbox.origin_y - pad_h)
            bw    = min(w - xmin, bbox.width  + pad_w * 2)
            bh    = min(h - ymin, bbox.height + pad_h * 2)
            cx    = xmin + bw // 2
            cy    = ymin + bh // 2

            matched_id = None
            min_dist   = TRACKER_TOLERANCE_PX

            for fid, (tx, ty) in face_tracker.items():
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist   = dist
                    matched_id = fid

            if matched_id is None:
                matched_id = next_face_id
                next_face_id += 1
                with scan_lock:
                    face_data[matched_id] = {"name": "Scanning...", "confidence": None}

            new_tracker[matched_id] = (cx, cy)

            with scan_lock:
                info       = face_data.get(matched_id, {"name": "Scanning...", "confidence": None})
                name       = info["name"]
                confidence = info["confidence"]

            if name == "Scanning...":
                color = (0, 165, 255)
            elif name == "Unknown":
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (xmin, ymin), (xmin + bw, ymin + bh), color, 2)
            
            lbl = f"{name}"
            if confidence is not None and name != "Unknown":
                lbl += f" ({int(confidence*100)}%)"
            cv2.putText(frame, lbl, (xmin, max(0, ymin - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

            faces_this_frame.append((matched_id, xmin, ymin, bw, bh))

    face_tracker = new_tracker

    # Jalankan Pengenalan Wajah DeepFace (Background Thread)
    if frame_count % SCAN_INTERVAL == 0 and faces_this_frame:

        def do_recognition(snap, faces):
            results_map = {}

            for fid, x, y, bw, bh in faces:
                if bw <= 0 or bh <= 0:
                    continue

                face_crop = snap[y:y + bh, x:x + bw]
                if face_crop.size == 0:
                    continue

                try:
                    # Mengubah detector_backend menjadi 'opencv' agar terjadi kalkulasi alignment posisi mata & hidung yang presisi
                    df_result = DeepFace.find(
                        img_path          = face_crop,
                        db_path           = DB_PATH,
                        model_name        = "ArcFace",
                        detector_backend  = "opencv",  
                        distance_metric   = "cosine",
                        enforce_detection = False,
                        silent            = True
                    )

                    if df_result and not df_result[0].empty:
                        row      = df_result[0].iloc[0]
                        path     = row["identity"]
                        dist_col = [c for c in df_result[0].columns if "distance" in c.lower()]
                        distance = float(row[dist_col[0]]) if dist_col else 0.99

                        nama = extract_name_from_path(path)
                        conf = distance_to_confidence(distance, threshold=ARCFACE_DIST_THRESHOLD)

                        print(f"[DEBUG AI] FaceID={fid} -> Terdeteksi: {nama} (Dist Cosine: {distance:.4f})")

                        if conf >= CONFIDENCE_THRESHOLD:
                            results_map[fid] = {"name": nama, "confidence": conf}
                        else:
                            results_map[fid] = {"name": "Unknown", "confidence": 0.0}
                    else:
                        results_map[fid] = {"name": "Unknown", "confidence": 0.0}

                except Exception as e:
                    print(f"[WARN AI] Error pengenalan FaceID={fid}: {e}")
                    results_map[fid] = {"name": "Unknown", "confidence": 0.0}

            with scan_lock:
                for fid, data in results_map.items():
                    if fid in face_data:
                        face_data[fid]["name"]       = data["name"]
                        face_data[fid]["confidence"] = data["confidence"]

        snap_copy = frame.copy()
        threading.Thread(
            target=do_recognition,
            args=(snap_copy, faces_this_frame),
            daemon=True
        ).start()

    cv2.imshow("Absensi Multi-Face AI", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_detector.close()
cv2.destroyAllWindows()