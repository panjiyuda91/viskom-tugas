import cv2
import numpy as np

# =============================================================================
# 1. KAMUS DATA RESISTOR
# =============================================================================
COLOR_CODES = {
    'Hitam':   {'digit': 0, 'multiplier': 1,          'tolerance': None},
    'Cokelat': {'digit': 1, 'multiplier': 10,         'tolerance': 1},
    'Merah':   {'digit': 2, 'multiplier': 100,        'tolerance': 2},
    'Oranye':  {'digit': 3, 'multiplier': 1000,       'tolerance': None},
    'Kuning':  {'digit': 4, 'multiplier': 10000,      'tolerance': None},
    'Hijau':   {'digit': 5, 'multiplier': 100000,     'tolerance': 0.5},
    'Biru':    {'digit': 6, 'multiplier': 1000000,    'tolerance': 0.25},
    'Ungu':    {'digit': 7, 'multiplier': 10000000,   'tolerance': 0.1},
    'Abu-abu': {'digit': 8, 'multiplier': 100000000,  'tolerance': 0.05},
    'Putih':   {'digit': 9, 'multiplier': 1000000000, 'tolerance': None},
    'Emas':    {'digit': None, 'multiplier': 0.1,     'tolerance': 5},
    'Perak':   {'digit': None, 'multiplier': 0.01,    'tolerance': 10}
}

# =============================================================================
# 2. DEFINISI RENTANG WARNA HSV
#    FIX #1: Merah sekarang punya DUA rentang (low + high) karena hue merah
#            "membungkus" di ujung spektrum HSV (dekat 0° dan dekat 180°).
#    FIX #2: Emas dan Perak dipisah lebih ketat dari Kuning dan Abu-abu.
#    FIX #3: Cokelat saturasi minimum dinaikkan agar tidak overlap dengan bodi.
#    Format: list of tuples [(lower1, upper1), (lower2, upper2), ...]
# =============================================================================
HSV_RANGES = {
    'Hitam':   [((0,   0,   0),   (180, 255,  55))],
    'Cokelat': [((5,  120,  30),  (18,  255, 140))],
    # --- FIX #1: Merah membutuhkan dua rentang ---
    'Merah':   [((0,   120,  80),  (8,   255, 255)),
                ((170, 120,  80),  (180, 255, 255))],
    'Oranye':  [((9,   180, 130),  (18,  255, 255))],
    # --- FIX #2: Kuning saturasi tinggi, pisahkan dari Emas ---
    'Kuning':  [((19,  180, 150),  (34,  255, 255))],
    'Hijau':   [((35,  100, 100),  (85,  255, 255))],
    'Biru':    [((86,  100, 100),  (125, 255, 255))],
    'Ungu':    [((126, 100,  80),  (155, 255, 255))],
    # --- FIX #2: Abu-abu saturasi sangat rendah, pisahkan dari Perak ---
    'Abu-abu': [((0,   0,   55),   (180,  40, 155))],
    'Putih':   [((0,   0,   190),  (180,  35, 255))],
    # --- FIX #2: Emas = saturasi sedang, value sedang (bukan saturasi tinggi) ---
    'Emas':    [((18,  60,  100),  (32,  160, 210))],
    # --- FIX #2: Perak = saturasi sangat rendah, value menengah-tinggi ---
    'Perak':   [((0,   0,   155),  (180,  45, 210))]
}

# =============================================================================
# 3. FUNGSI FORMAT NILAI RESISTANSI
# =============================================================================
def format_resistance(value):
    """Mengubah nilai Ohm ke notasi yang lebih mudah dibaca (kΩ, MΩ, dll)."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3g} MΩ"
    elif value >= 1_000:
        return f"{value / 1_000:.3g} kΩ"
    else:
        return f"{value:.3g} Ω"

# =============================================================================
# 4. FUNGSI HITUNG RESISTANSI
#    FIX #3: Toleransi None sekarang ditampilkan sebagai "±?" bukan crash/None
# =============================================================================
def calculate_resistance(bands):
    """Menghitung nilai resistansi berdasarkan urutan warna pita."""
    if len(bands) < 3:
        return "Gagal: Pita warna tidak cukup terdeteksi."

    try:
        if len(bands) == 4:
            digit1     = COLOR_CODES[bands[0]]['digit']
            digit2     = COLOR_CODES[bands[1]]['digit']
            multiplier = COLOR_CODES[bands[2]]['multiplier']
            tolerance  = COLOR_CODES[bands[3]]['tolerance']
            if digit1 is None or digit2 is None:
                return "Gagal: Warna pita berada di posisi yang tidak valid."
            value = ((digit1 * 10) + digit2) * multiplier

        elif len(bands) >= 5:
            digit1     = COLOR_CODES[bands[0]]['digit']
            digit2     = COLOR_CODES[bands[1]]['digit']
            digit3     = COLOR_CODES[bands[2]]['digit']
            multiplier = COLOR_CODES[bands[3]]['multiplier']
            tolerance  = COLOR_CODES[bands[4]]['tolerance']
            if digit1 is None or digit2 is None or digit3 is None:
                return "Gagal: Warna pita berada di posisi yang tidak valid."
            value = ((digit1 * 100) + (digit2 * 10) + digit3) * multiplier

        else:
            return "Format pita tidak dikenali."

        # FIX #3: Tangani toleransi None dengan graceful
        tol_str = f"±{tolerance}%" if tolerance is not None else "±?"
        return f"{format_resistance(value)} {tol_str}"

    except (TypeError, KeyError) as e:
        return f"Gagal menghitung: {e}"

# =============================================================================
# 5. FUNGSI DETEKSI REGION RESISTOR (BARU)
#    Mengisolasi area badan resistor agar deteksi pita tidak terganggu
#    oleh warna latar belakang yang serupa.
# =============================================================================
def get_resistor_roi(img):
    """
    Mendeteksi badan resistor dan mengembalikan bounding box ROI.
    Strategi: temukan objek terbesar yang bukan latar belakang.
    """
    h_img, w_img = img.shape[:2]

    # Konversi ke grayscale lalu threshold untuk segmentasi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Coba deteksi dengan Canny edge
    edges = cv2.Canny(gray, 30, 100)
    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_roi = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:  # Abaikan kontur sangat kecil
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0

        # Resistor memanjang secara horizontal: lebar jauh lebih besar dari tinggi
        if aspect > 2.0 and area > best_area:
            best_area = area
            best_roi = (x, y, w, h)

    if best_roi is None:
        # Fallback: gunakan strip horizontal tengah gambar
        margin_x = int(w_img * 0.05)
        margin_y = int(h_img * 0.25)
        best_roi = (margin_x, margin_y, w_img - 2 * margin_x, h_img - 2 * margin_y)

    return best_roi

# =============================================================================
# 6. FUNGSI UTAMA PEMROSESAN GAMBAR
# =============================================================================
def process_resistor_image(image_path):
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"Error: Gambar '{image_path}' tidak ditemukan.")
        return

    # Resize standar
    img = cv2.resize(img_original, (800, 500))
    display = img.copy()

    # --- FIX #4: Isolasi ROI resistor terlebih dahulu ---
    roi_x, roi_y, roi_w, roi_h = get_resistor_roi(img)

    # Tambahkan sedikit padding vertikal agar seluruh pita masuk ROI
    pad = 10
    roi_y1 = max(0, roi_y - pad)
    roi_y2 = min(img.shape[0], roi_y + roi_h + pad)
    roi_x1 = max(0, roi_x)
    roi_x2 = min(img.shape[1], roi_x + roi_w)

    roi_img = img[roi_y1:roi_y2, roi_x1:roi_x2]

    # Terapkan Gaussian Blur hanya pada ROI
    blurred = cv2.GaussianBlur(roi_img, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    detected_bands = []
    roi_h_px = roi_y2 - roi_y1

    for color_name, ranges in HSV_RANGES.items():
        # Gabungkan semua mask untuk warna ini (penting untuk Merah)
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for (lower, upper) in ranges:
            lower_np = np.array(lower, dtype=np.uint8)
            upper_np = np.array(upper, dtype=np.uint8)
            combined_mask = cv2.bitwise_or(combined_mask, cv2.inRange(hsv, lower_np, upper_np))

        # Operasi morfologi untuk bersihkan noise
        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # --- FIX #5: Filter geometris lebih fleksibel ---
            # Pita harus merentang setidaknya 30% tinggi ROI (memanjang vertikal)
            # DAN tidak terlalu lebar (bukan badan resistor)
            height_ratio = h / roi_h_px
            aspect_ok = h > w  # Lebih tinggi dari lebar
            tall_enough = height_ratio > 0.25
            not_too_wide = w < roi_w * 0.25  # Tidak lebih dari 25% lebar ROI

            if not (aspect_ok and tall_enough and not_too_wide):
                continue

            # --- Filter tumpang tindih (NMS) ---
            # Normalisasi threshold overlap ke pixel aktual
            overlap_thresh = max(15, int(roi_w * 0.03))
            is_overlapping = False
            for b in detected_bands:
                if abs(x - b['x']) < overlap_thresh:
                    # Jika overlap, pertahankan yang area-nya lebih besar
                    if area > b['area']:
                        detected_bands.remove(b)
                    else:
                        is_overlapping = True
                    break

            if not is_overlapping:
                # Koordinat global (relatif ke gambar penuh)
                gx = roi_x1 + x
                gy = roi_y1 + y
                detected_bands.append({
                    'color': color_name,
                    'x': x,       # Koordinat lokal ROI untuk sorting
                    'gx': gx,     # Koordinat global untuk gambar
                    'gy': gy,
                    'w': w,
                    'h': h,
                    'area': area
                })

    # Urutkan pita kiri ke kanan
    detected_bands = sorted(detected_bands, key=lambda b: b['x'])
    band_colors = [b['color'] for b in detected_bands]

    print(f"Pita terdeteksi (Kiri -> Kanan): {band_colors}")

    # Hitung resistansi
    result = calculate_resistance(band_colors)
    print(f"Nilai Resistor: {result}")

    # --- Gambar visualisasi ---
    # Kotak ROI resistor (biru muda)
    cv2.rectangle(display, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 200, 0), 1)

    # Kotak setiap pita yang terdeteksi
    BAND_COLORS_BGR = {
        'Hitam': (50, 50, 50), 'Cokelat': (19, 69, 139), 'Merah': (0, 0, 220),
        'Oranye': (0, 165, 255), 'Kuning': (0, 220, 220), 'Hijau': (0, 180, 0),
        'Biru': (220, 100, 0), 'Ungu': (180, 0, 180), 'Abu-abu': (128, 128, 128),
        'Putih': (240, 240, 240), 'Emas': (0, 215, 255), 'Perak': (192, 192, 192)
    }
    for i, b in enumerate(detected_bands):
        box_color = BAND_COLORS_BGR.get(b['color'], (0, 255, 0))
        cv2.rectangle(display, (b['gx'], b['gy']), (b['gx'] + b['w'], b['gy'] + b['h']), box_color, 2)
        label = f"{i+1}:{b['color']}"
        cv2.putText(display, label, (b['gx'], b['gy'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(display, label, (b['gx'], b['gy'] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    # Teks hasil di bagian atas
    result_text = f"Hasil: {result}"
    text_color = (0, 200, 0) if "Ohm" in result or "Ω" in result else (0, 0, 220)
    cv2.putText(display, result_text, (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 3)

    cv2.imshow("Deteksi Resistor", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =============================================================================
# 7. ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import sys
    image_file = sys.argv[1] if len(sys.argv) > 1 else "resistor3.jpg"
    process_resistor_image(image_file)