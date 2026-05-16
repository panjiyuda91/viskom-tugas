"""
Resistor Color Band Detector v3
================================
Hybrid approach:
  1. Find resistor body ROI via edge detection
  2. Extract center strip, estimate body color
  3. Build color-change profile → find band positions
  4. For each band segment: classify using HSV rules first,
     fall back to LAB nearest-neighbor
  5. Merge duplicates, calculate resistance

Run:  python resistor_detector.py image.png [--debug]
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
import sys

# =============================================================================
# 1. RESISTOR DATA
# =============================================================================
COLOR_CODES = {
    'Hitam':   {'digit': 0, 'multiplier': 1,           'tolerance': None},
    'Cokelat': {'digit': 1, 'multiplier': 10,          'tolerance': 1},
    'Merah':   {'digit': 2, 'multiplier': 100,         'tolerance': 2},
    'Oranye':  {'digit': 3, 'multiplier': 1000,        'tolerance': None},
    'Kuning':  {'digit': 4, 'multiplier': 10000,       'tolerance': None},
    'Hijau':   {'digit': 5, 'multiplier': 100000,      'tolerance': 0.5},
    'Biru':    {'digit': 6, 'multiplier': 1000000,     'tolerance': 0.25},
    'Ungu':    {'digit': 7, 'multiplier': 10000000,    'tolerance': 0.1},
    'Abu-abu': {'digit': 8, 'multiplier': 100000000,   'tolerance': 0.05},
    'Putih':   {'digit': 9, 'multiplier': 1000000000,  'tolerance': None},
    'Emas':    {'digit': None, 'multiplier': 0.1,      'tolerance': 5},
    'Perak':   {'digit': None, 'multiplier': 0.01,     'tolerance': 10},
}

# =============================================================================
# 2. HSV-BASED CLASSIFICATION (primary method)
#    Each entry: list of (lower_hsv, upper_hsv) tuples
#    H: 0-179, S: 0-255, V: 0-255
# =============================================================================
HSV_RULES = {
    # Very dark pixels regardless of hue
    'Hitam':   [((0,   0,   0),   (180, 255,  60))],
    # Very bright, very low saturation
    'Putih':   [((0,   0,  200),  (180,  40, 255))],
    # Low saturation mid-value = gray
    'Abu-abu': [((0,   0,  61),   (180,  50, 199))],
    # Silver: low saturation, higher value than gray
    'Perak':   [((0,   0,  160),  (180,  45, 210))],
    # Red wraps around 0°/180°
    'Merah':   [((0,  120,  60),  (9,   255, 255)),
                ((170,120,  60),  (180, 255, 255))],
    # Orange: narrow hue band, high saturation
    'Oranye':  [((9,  160, 100),  (20,  255, 255))],
    # Yellow: vivid, high saturation
    'Kuning':  [((20, 160, 150),  (35,  255, 255))],
    # Gold: similar hue to yellow but lower saturation & value
    'Emas':    [((15,  60,  80),  (35,  159, 220))],
    # Brown: low-value orange
    'Cokelat': [((5,  100,  20),  (20,  255, 130))],
    # Green
    'Hijau':   [((36, 100,  60),  (85,  255, 255))],
    # Blue
    'Biru':    [((86, 100,  60),  (125, 255, 255))],
    # Purple/Violet
    'Ungu':    [((126, 80,  60),  (155, 255, 255))],
}

# Priority order matters: more specific rules first
HSV_PRIORITY = [
    'Hitam', 'Putih', 'Merah', 'Ungu', 'Biru',
    'Hijau', 'Oranye', 'Cokelat', 'Kuning', 'Emas',
    'Abu-abu', 'Perak'
]

def classify_pixel_hsv(bgr_pixel):
    """
    Classify a single BGR pixel using HSV rules.
    Returns (color_name, confidence) where confidence = fraction of
    pixels in a tiny patch that match. Returns None if no rule matches.
    """
    px = np.uint8([[bgr_pixel]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0]

    for name in HSV_PRIORITY:
        ranges = HSV_RULES[name]
        for (lo, hi) in ranges:
            lo_arr = np.array(lo, dtype=np.uint8)
            hi_arr = np.array(hi, dtype=np.uint8)
            if cv2.inRange(np.uint8([[hsv]]), lo_arr, hi_arr)[0][0] == 255:
                return name
    return None  # Unclassified


def classify_pixels_hsv_voting(bgr_pixels):
    """
    Given an array of BGR pixels, vote on the most common HSV classification.
    Returns (winner_name, vote_fraction).
    """
    votes = {}
    total = len(bgr_pixels)
    for px in bgr_pixels:
        name = classify_pixel_hsv(px.tolist())
        if name:
            votes[name] = votes.get(name, 0) + 1

    if not votes:
        return None, 0.0

    winner = max(votes, key=votes.get)
    return winner, votes[winner] / total

# =============================================================================
# 3. BODY DETECTION & ROI
# =============================================================================
def find_resistor_roi(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 20, 80)
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_score = None, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / (bh + 1e-5)
        if aspect > 1.5 and bw > w * 0.2:
            score = area * min(aspect, 8.0)
            if score > best_score:
                best_score = score
                best = (x, y, x + bw, y + bh)

    if best is None:
        mx, my = int(w * 0.05), int(h * 0.2)
        best = (mx, my, w - mx, h - my)
    return best


def estimate_body_color_lab(strip):
    """Body color = median of entire strip (body dominates)."""
    pixels = strip.reshape(-1, 3)
    median_bgr = np.median(pixels, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(np.uint8([[median_bgr]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
    return median_bgr, lab

# =============================================================================
# 4. PROFILE-BASED BAND DETECTION
# =============================================================================
def compute_color_change_profile(strip, window=4):
    w = strip.shape[1]
    profile = np.zeros(w, dtype=np.float32)
    for x in range(window, w - window):
        left_med  = np.median(strip[:, x-window:x].reshape(-1,3), axis=0)
        right_med = np.median(strip[:, x:x+window].reshape(-1,3), axis=0)
        l = cv2.cvtColor(np.uint8([[left_med]]),  cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        r = cv2.cvtColor(np.uint8([[right_med]]), cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        profile[x] = np.linalg.norm(l - r)
    return profile


def find_band_boundaries(profile, strip_w, min_ratio=0.03, min_height=6.0):
    smooth_k = max(3, strip_w // 50)
    smoothed = np.convolve(profile, np.ones(smooth_k)/smooth_k, mode='same')
    min_dist = max(4, int(strip_w * min_ratio))
    peaks, _ = find_peaks(smoothed, height=min_height, distance=min_dist, prominence=3.0)

    if len(peaks) < 2:
        # Try with lower threshold
        peaks, _ = find_peaks(smoothed, height=min_height*0.5, distance=min_dist, prominence=1.5)

    boundaries = sorted(set([0] + list(peaks) + [strip_w]))
    segments = []
    for i in range(len(boundaries)-1):
        w_seg = boundaries[i+1] - boundaries[i]
        if w_seg >= int(strip_w * min_ratio):
            segments.append((boundaries[i], boundaries[i+1]))

    return segments, smoothed

# =============================================================================
# 5. SEGMENT COLOR CLASSIFICATION
# =============================================================================
def classify_segment(strip, x1, x2, body_lab, body_thresh=20.0, debug=False):
    """
    Extract pixels from segment, filter body color, then classify via HSV voting.
    """
    col_bgr = strip[:, x1:x2].reshape(-1, 3)
    if len(col_bgr) == 0:
        return None, 0.0

    # Filter out body-colored pixels using LAB distance
    non_body = []
    for px in col_bgr:
        px_u8 = np.uint8([[px]])
        px_lab = cv2.cvtColor(px_u8, cv2.COLOR_BGR2LAB)[0][0].astype(np.float32)
        if np.linalg.norm(px_lab - body_lab) > body_thresh:
            non_body.append(px)

    body_ratio = 1.0 - (len(non_body) / len(col_bgr))

    if debug:
        med = np.median(col_bgr, axis=0).astype(int)
        print(f"    Seg [{x1}-{x2}]: {len(non_body)}/{len(col_bgr)} non-body px, "
              f"body_ratio={body_ratio:.2f}, median_bgr={med}")

    # If >70% of pixels are body color, this IS the body — skip
    if body_ratio > 0.70:
        return None, body_ratio

    # Use all pixels (not just non-body) for HSV voting — body filter already handled
    target_pixels = np.array(non_body) if len(non_body) >= 5 else col_bgr
    color, confidence = classify_pixels_hsv_voting(target_pixels)
    return color, confidence

# =============================================================================
# 6. RESISTANCE CALCULATION
# =============================================================================
def format_resistance(value):
    if value >= 1_000_000:
        return f"{value/1_000_000:.3g} MΩ"
    elif value >= 1_000:
        return f"{value/1_000:.3g} kΩ"
    else:
        return f"{value:.3g} Ω"

def calculate_resistance(bands):
    if len(bands) < 3:
        return f"Gagal: Hanya {len(bands)} pita terdeteksi (butuh minimal 3)."
    try:
        if len(bands) == 3:
            d1, d2 = COLOR_CODES[bands[0]]['digit'], COLOR_CODES[bands[1]]['digit']
            mul, tol = COLOR_CODES[bands[2]]['multiplier'], 20
            if d1 is None or d2 is None: return "Gagal: Digit tidak valid."
            value = ((d1*10)+d2)*mul
        elif len(bands) == 4:
            d1, d2 = COLOR_CODES[bands[0]]['digit'], COLOR_CODES[bands[1]]['digit']
            mul = COLOR_CODES[bands[2]]['multiplier']
            tol = COLOR_CODES[bands[3]]['tolerance']
            if d1 is None or d2 is None: return "Gagal: Digit tidak valid."
            value = ((d1*10)+d2)*mul
        else:  # 5+ bands
            d1 = COLOR_CODES[bands[0]]['digit']
            d2 = COLOR_CODES[bands[1]]['digit']
            d3 = COLOR_CODES[bands[2]]['digit']
            mul = COLOR_CODES[bands[3]]['multiplier']
            tol = COLOR_CODES[bands[4]]['tolerance']
            if d1 is None or d2 is None or d3 is None: return "Gagal: Digit tidak valid."
            value = ((d1*100)+(d2*10)+d3)*mul
        tol_str = f"±{tol}%" if tol is not None else "±?"
        return f"{format_resistance(value)} {tol_str}"
    except (TypeError, KeyError) as e:
        return f"Gagal: {e}"

# =============================================================================
# 7. MAIN
# =============================================================================
def process_resistor_image(image_path, debug=False):
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: '{image_path}' tidak ditemukan.")
        return

    img = cv2.resize(img_orig, (900, 550))
    display = img.copy()
    h_img, w_img = img.shape[:2]

    # ── Step 1: Find resistor body ──────────────────────────────────────────
    x1, y1, x2, y2 = find_resistor_roi(img)
    if debug:
        print(f"ROI: ({x1},{y1})->({x2},{y2})")

    # ── Step 2: Extract center strip ────────────────────────────────────────
    cy = (y1 + y2) // 2
    sh = max(12, (y2 - y1) // 6)   # ±1/6 of body height
    sy1, sy2 = max(0, cy-sh), min(h_img, cy+sh)

    # Trim horizontal margins (wire leads)
    margin = int((x2 - x1) * 0.06)
    sx1, sx2 = x1 + margin, x2 - margin
    strip   = img[sy1:sy2, sx1:sx2].copy()
    strip_w = sx2 - sx1

    # ── Step 3: Estimate body color ──────────────────────────────────────────
    body_bgr, body_lab = estimate_body_color_lab(strip)
    if debug:
        print(f"Body BGR: {body_bgr}")

    # ── Step 4: Color-change profile ─────────────────────────────────────────
    profile = compute_color_change_profile(strip)
    segments, smoothed = find_band_boundaries(profile, strip_w)
    if debug:
        print(f"Segmen: {segments}")

    # ── Step 5: Classify segments ─────────────────────────────────────────────
    raw_bands = []
    for (seg_x1, seg_x2) in segments:
        color, conf = classify_segment(strip, seg_x1, seg_x2, body_lab,
                                       body_thresh=20.0, debug=debug)
        if color is None:
            if debug:
                print(f"  Seg [{seg_x1}-{seg_x2}]: SKIP (body)")
            continue
        if debug:
            print(f"  Seg [{seg_x1}-{seg_x2}]: {color} (conf={conf:.2f})")
        raw_bands.append({
            'color': color,
            'conf':  conf,
            'gx1':   sx1 + seg_x1,
            'gx2':   sx1 + seg_x2,
        })

    # ── Step 6: Merge adjacent same-color bands ───────────────────────────────
    merged = []
    for b in raw_bands:
        if merged and merged[-1]['color'] == b['color']:
            merged[-1]['gx2'] = b['gx2']
        else:
            merged.append(dict(b))

    band_colors = [b['color'] for b in merged]
    print(f"\nPita terdeteksi (Kiri -> Kanan): {band_colors}")
    result = calculate_resistance(band_colors)
    print(f"Nilai Resistor       : {result}\n")

    # ── Step 7: Draw ──────────────────────────────────────────────────────────
    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 180, 0), 1)
    cv2.rectangle(display, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)

    VIZ = {
        'Hitam':(50,50,50),    'Cokelat':(30,105,210),  'Merah':(0,30,230),
        'Oranye':(0,130,255),  'Kuning':(0,220,220),    'Hijau':(0,210,30),
        'Biru':(230,90,0),     'Ungu':(200,0,200),      'Abu-abu':(140,140,140),
        'Putih':(240,240,240), 'Emas':(0,200,255),      'Perak':(180,180,200),
    }
    for i, b in enumerate(merged):
        col = VIZ.get(b['color'], (0,255,0))
        cv2.rectangle(display, (b['gx1'], sy1-6), (b['gx2'], sy2+6), col, 2)
        label = f"{i+1}:{b['color']}"
        cv2.putText(display, label, (b['gx1'], sy1-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 3)
        cv2.putText(display, label, (b['gx1'], sy1-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

    ok = 'Ω' in result
    tc = (20, 200, 20) if ok else (20, 20, 220)
    cv2.putText(display, f"Hasil: {result}", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 5)
    cv2.putText(display, f"Hasil: {result}", (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, tc, 2)

    if debug:
        # Show color-change profile as a chart
        ph = 80
        prof_vis = np.zeros((ph, strip_w, 3), dtype=np.uint8)
        mv = smoothed.max() or 1
        for x in range(strip_w):
            bh = int((smoothed[x]/mv)*ph)
            cv2.line(prof_vis, (x,ph),(x,ph-bh),(0,220,100),1)
        for (sx, ex) in segments:
            cv2.line(prof_vis,(sx,0),(sx,ph),(0,100,255),1)
        cv2.imshow("Profile", prof_vis)

    cv2.imshow("Deteksi Resistor", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "resistor1.png"
    dbg = "--debug" in sys.argv
    process_resistor_image(f, debug=dbg)