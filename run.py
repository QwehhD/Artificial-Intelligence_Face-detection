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
# KONFIGURASI
# ============================================================
DB_PATH = "dataset"
SCAN_INTERVAL = 20          # Scan setiap 20 frame (~0.67 detik pada 30fps)
VOTE_ROUNDS = 3             # Jumlah scan per wajah untuk voting konsensus
CONFIDENCE_THRESHOLD = 0.45 # Jika confidence < 45%, tandai Unknown
TRACKER_TOLERANCE_PX = 80   # Batas toleransi pergeseran piksel centroid tracker
BBOX_PAD_RATIO = 0.20       # Padding bbox 20% untuk tangkap dahi/rahang

# ============================================================
# DOWNLOAD MODEL MEDIAPIPE
# Coba full_range dulu (lebih presisi), fallback ke short_range
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
        print(f"[INFO] Menggunakan model lokal: {_path} ({_label})")
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
    raise RuntimeError(
        "[ERROR] Semua model gagal diunduh.\n"
        "Unduh manual salah satu file berikut lalu letakkan di folder yang sama dengan script:\n"
        "  https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    )

# ============================================================
# INISIALISASI MEDIAPIPE — Full Range, confidence lebih tinggi
# ============================================================
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.75   # Dinaikkan dari 0.6 → lebih selektif
)
face_detector = mp_vision.FaceDetector.create_from_options(options)

# ============================================================
# INISIALISASI KAMERA
# ============================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # Matikan auto-exposure
cap.set(cv2.CAP_PROP_EXPOSURE, 0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)          # Aktifkan autofocus

print(f"[INFO] Exposure kamera: {cap.get(cv2.CAP_PROP_EXPOSURE)}")

# ============================================================
# HELPER — Normalisasi Cahaya (CLAHE + White Balance sederhana)
# ============================================================
def normalize_face(face_img: np.ndarray) -> np.ndarray:
    """Terapkan CLAHE + koreksi white balance sederhana."""
    try:
        # --- White Balance (Gray World) ---
        b, g, r = cv2.split(face_img.astype(np.float32))
        b_mean, g_mean, r_mean = b.mean(), g.mean(), r.mean()
        gray_mean = (b_mean + g_mean + r_mean) / 3.0
        face_img = cv2.merge([
            np.clip(b * (gray_mean / (b_mean + 1e-6)), 0, 255).astype(np.uint8),
            np.clip(g * (gray_mean / (g_mean + 1e-6)), 0, 255).astype(np.uint8),
            np.clip(r * (gray_mean / (r_mean + 1e-6)), 0, 255).astype(np.uint8),
        ])

        # --- CLAHE pada kanal Y (luminance) ---
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        channels = list(cv2.split(ycrcb))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels[0] = clahe.apply(channels[0])
        face_img = cv2.cvtColor(cv2.merge(channels), cv2.COLOR_YCrCb2BGR)
    except Exception as e:
        print(f"[WARN] Normalisasi gagal: {e}")
    return face_img


# ============================================================
# HELPER — Hitung Confidence dari jarak DeepFace
# ============================================================
def distance_to_confidence(distance: float, threshold: float = 0.6) -> float:
    """
    Konversi jarak embedding ke persentase keyakinan.
    distance=0.0  → 100% (identik sempurna)
    distance≥threshold → ~0%
    Menggunakan fungsi sigmoid terbalik yang diskalakan.
    """
    if distance <= 0.0:
        return 1.0
    if distance >= threshold:
        return 0.0
    # Normalisasi ke [0, 1] lalu terapkan kurva kuadratik agar lebih intuitif
    ratio = distance / threshold          # 0.0 = paling mirip, 1.0 = batas
    confidence = (1.0 - ratio) ** 1.5    # Kurva sedikit agresif di batas
    return round(min(max(confidence, 0.0), 1.0), 4)


# ============================================================
# HELPER — Gambar Label Bergradien di atas BBox
# ============================================================
def draw_label(frame, text, x, y, color, confidence=None):
    """Gambar background semi-transparan + teks label dengan confidence."""
    label = text
    if confidence is not None:
        label = f"{text}  {int(confidence * 100)}%"

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.65
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 5

    # Background semi-transparan
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y - th - pad * 2 - baseline),
        (x + tw + pad, y + baseline),
        (30, 30, 30), -1
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Teks utama
    cv2.putText(frame, text, (x, y - baseline - 1), font, font_scale, color, thickness, cv2.LINE_AA)

    # Teks persentase (warna berbeda untuk keterbacaan)
    if confidence is not None:
        conf_text = f"  {int(confidence * 100)}%"
        (name_w, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        pct_color = (
            (100, 255, 100) if confidence >= 0.70 else
            (100, 200, 255) if confidence >= 0.50 else
            (80, 80, 255)
        )
        cv2.putText(
            frame, conf_text,
            (x + name_w, y - baseline - 1),
            font, font_scale, pct_color, thickness, cv2.LINE_AA
        )


# ============================================================
# STRUKTUR DATA TRACKING
# ============================================================
face_tracker = {}   # {face_id: (cx, cy)}
face_data   = {}    # {face_id: {"name": str, "confidence": float, "votes": []}}
next_face_id = 0
scan_lock = threading.Lock()
frame_count = 0

print("\n=== SISTEM ABSENSI MULTI-FACE AI (PRESISI TINGGI + CONFIDENCE) ===")
print("Tekan [Q] untuk Keluar\n")

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape
    faces_this_frame = []

    # --------------------------------------------------------
    # DETEKSI WAJAH (setiap frame)
    # --------------------------------------------------------
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result    = face_detector.detect(mp_image)

    new_tracker = {}

    if result.detections:
        for detection in result.detections:
            bbox = detection.bounding_box
            det_score = detection.categories[0].score if detection.categories else 0.0

            # Abaikan deteksi berkualitas rendah
            if det_score < 0.75:
                continue

            # --- Padding bbox ---
            pad_w = int(bbox.width  * BBOX_PAD_RATIO)
            pad_h = int(bbox.height * BBOX_PAD_RATIO)
            xmin  = max(0, bbox.origin_x - pad_w)
            ymin  = max(0, bbox.origin_y - pad_h)
            bw    = min(w - xmin, bbox.width  + pad_w * 2)
            bh    = min(h - ymin, bbox.height + pad_h * 2)

            cx = xmin + bw // 2
            cy = ymin + bh // 2

            # --- Centroid Tracking ---
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
                    face_data[matched_id] = {
                        "name": "Scanning...",
                        "confidence": None,
                        "votes": []
                    }

            new_tracker[matched_id] = (cx, cy)

            with scan_lock:
                info       = face_data.get(matched_id, {"name": "Scanning...", "confidence": None})
                name       = info["name"]
                confidence = info["confidence"]

            # --- Pilih warna bbox ---
            if name == "Scanning...":
                color = (0, 165, 255)   # Oranye
            elif name == "Unknown":
                color = (0, 0, 255)     # Merah
            else:
                # Gradasi hijau berdasarkan confidence
                green_level = int(100 + 155 * (confidence or 0))
                color = (0, green_level, 50)

            # --- Gambar BBox ---
            cv2.rectangle(frame, (xmin, ymin), (xmin + bw, ymin + bh), color, 2)

            # --- Gambar Label + Confidence ---
            draw_label(
                frame, name, xmin, ymin,
                color,
                confidence if (name not in ("Scanning...", "Unknown")) else None
            )

            # --- Tambahkan keypoint (mata, hidung, mulut) ---
            for kp in detection.keypoints:
                kp_x = int(kp.x * w)
                kp_y = int(kp.y * h)
                cv2.circle(frame, (kp_x, kp_y), 3, (255, 200, 0), -1)

            faces_this_frame.append((matched_id, xmin, ymin, bw, bh))

    face_tracker = new_tracker

    # --------------------------------------------------------
    # REKOGNISI DEEPFACE (interval + background thread)
    # --------------------------------------------------------
    if frame_count % SCAN_INTERVAL == 0 and faces_this_frame:

        def do_recognition(snap, faces):
            """
            Jalankan VOTE_ROUNDS kali DeepFace.find per wajah,
            ambil konsensus nama + rata-rata confidence.
            """
            results_map = {}  # {fid: {"name": str, "confidence": float}}

            for fid, x, y, bw, bh in faces:
                if bw <= 0 or bh <= 0:
                    continue

                face_crop = snap[y:y + bh, x:x + bw]
                if face_crop.size == 0:
                    continue

                face_crop = normalize_face(face_crop)

                vote_names  = []
                vote_scores = []

                for _ in range(VOTE_ROUNDS):
                    try:
                        df_result = DeepFace.find(
                            img_path=face_crop,
                            db_path=DB_PATH,
                            model_name="ArcFace",        # ArcFace — lebih akurat dari VGG-Face
                            detector_backend="skip",     # Deteksi sudah dilakukan MediaPipe
                            distance_metric="cosine",    # Cosine lebih stabil
                            enforce_detection=False,
                            silent=True
                        )

                        if df_result and not df_result[0].empty:
                            row      = df_result[0].iloc[0]
                            path     = row["identity"]
                            # Kolom jarak: ArcFace cosine → "ArcFace_cosine"
                            dist_col = [c for c in df_result[0].columns if "distance" in c.lower()]
                            distance = float(row[dist_col[0]]) if dist_col else 0.5

                            conf = distance_to_confidence(distance, threshold=0.5)

                            if conf >= CONFIDENCE_THRESHOLD:
                                nama = os.path.splitext(os.path.basename(path))[0].upper()
                                vote_names.append(nama)
                                vote_scores.append(conf)
                            else:
                                vote_names.append("Unknown")
                                vote_scores.append(0.0)
                        else:
                            vote_names.append("Unknown")
                            vote_scores.append(0.0)

                    except Exception as e:
                        vote_names.append("Unknown")
                        vote_scores.append(0.0)

                # --- Ambil nama terbanyak (majority vote) ---
                if vote_names:
                    from collections import Counter
                    majority_name = Counter(vote_names).most_common(1)[0][0]
                    # Confidence = rata-rata hanya dari vote yang setuju
                    agree_scores = [
                        s for n, s in zip(vote_names, vote_scores)
                        if n == majority_name
                    ]
                    avg_conf = sum(agree_scores) / len(agree_scores) if agree_scores else 0.0
                    results_map[fid] = {"name": majority_name, "confidence": avg_conf}
                else:
                    results_map[fid] = {"name": "Unknown", "confidence": 0.0}

            # --- Update global state ---
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

    # --------------------------------------------------------
    # INFO OVERLAY — jumlah wajah terdeteksi
    # --------------------------------------------------------
    face_count = len(face_tracker)
    cv2.putText(
        frame,
        f"Wajah Terdeteksi: {face_count}",
        (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (10, 58), cv2.FONT_HERSHEY_DUPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA
    )

    cv2.imshow("Absensi Multi-Face (Presisi Tinggi)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# CLEANUP
# ============================================================
cap.release()
face_detector.close()
cv2.destroyAllWindows()