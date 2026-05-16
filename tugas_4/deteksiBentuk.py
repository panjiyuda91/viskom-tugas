import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLUR_KERNEL  = (5, 5)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
CANNY_LOW    = 50
CANNY_HIGH   = 150


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------
def proses_awal(image_path):
    """
    Returns: (img_asli, img_gray, img_mask, img_blur_biner, img_blur_gray, img_canny)

    Kenapa ada dua hasil blur?
    --------------------------
    img_blur_biner : GaussianBlur diterapkan langsung ke img_mask (biner 0/255).
                     Hasilnya tetap terlihat seperti versi "soft" dari mask biner.
                     Ini yang DITAMPILKAN di window agar alur visualisasi konsisten:
                     biner → blur biner → canny dari blur biner.

    img_blur_gray  : GaussianBlur diterapkan ke grayscale yang sudah di-mask.
                     Ini HANYA dipakai oleh HoughCircles, karena algoritma tersebut
                     membutuhkan informasi gradien intensitas (tidak bisa dari biner).
                     Tidak ditampilkan sebagai window karena bukan bagian dari
                     alur visualisasi utama.

    img_canny      : Hasil Canny edge detection dari img_blur_biner.
                     Ditampilkan sebagai window tersendiri sesuai permintaan dosen.
                     Ini juga yang digunakan oleh deteksi_garis dan deteksi_elips
                     secara internal.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Gambar '{image_path}' tidak ditemukan!")
        return None, None, None, None, None, None

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # HSV saturation mask — biner (0 atau 255)
    img_mask = cv2.inRange(img_hsv, np.array([0, 60, 20]), np.array([180, 255, 255]))
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN,  MORPH_KERNEL)
    img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

    # Blur dari mask biner → untuk ditampilkan dan sebagai input Canny
    img_blur_biner = cv2.GaussianBlur(img_mask, BLUR_KERNEL, 0)

    # Blur dari grayscale termasked → KHUSUS untuk HoughCircles
    img_clean      = cv2.bitwise_and(img_gray, img_gray, mask=img_mask)
    img_blur_gray  = cv2.GaussianBlur(img_clean, BLUR_KERNEL, 0)

    # Canny edge dari blur biner
    img_canny = cv2.Canny(img_blur_biner, CANNY_LOW, CANNY_HIGH)

    return img, img_gray, img_mask, img_blur_biner, img_blur_gray, img_canny


# ---------------------------------------------------------------------------
# Line detection
# ---------------------------------------------------------------------------
def deteksi_garis(img_mask):
    """
    approxPolyDP-based — menggunakan mask biner langsung (bukan blur).
    Dual-signal circle exclusion: circularity > 0.82 AND vertex_count > 7.
    """
    contours, _ = cv2.findContours(
        img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    all_lines = []
    for cnt in contours:
        area      = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or area < 500:
            continue

        circularity  = (4 * np.pi * area) / (perimeter ** 2)
        epsilon      = 0.018 * perimeter
        approx       = cv2.approxPolyDP(cnt, epsilon, True)
        vertex_count = len(approx)

        if circularity > 0.82 and vertex_count > 7:
            continue

        n = len(approx)
        for i in range(n):
            x1, y1 = approx[i][0]
            x2, y2 = approx[(i + 1) % n][0]
            all_lines.append(np.array([[x1, y1, x2, y2]]))

    return all_lines if all_lines else None


# ---------------------------------------------------------------------------
# Circle detection
# ---------------------------------------------------------------------------
def deteksi_lingkaran(img_blur_gray):
    """
    Menggunakan img_blur_gray (grayscale termasked) — bukan img_blur_biner.
    HoughCircles bekerja dari akumulator gradien orientasi, sehingga butuh
    informasi intensitas bertingkat (grayscale), bukan tepi biner.
    """
    circles = cv2.HoughCircles(
        img_blur_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=100, param2=28, minRadius=50, maxRadius=160
    )
    if circles is not None:
        circles = np.int32(np.around(circles))
    return circles


# ---------------------------------------------------------------------------
# Ellipse detection
# ---------------------------------------------------------------------------
def deteksi_elips(img_asli, img_mask, circles):
    """
    Cross-reference dengan circles untuk menghindari lingkaran terdeteksi
    sebagai elips. Menggunakan img_mask biner langsung.
    """
    img_h, img_w = img_asli.shape[:2]
    min_area = (img_h * img_w) * 0.001

    contours, _ = cv2.findContours(
        img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    valid_ellipses = []
    for cnt in contours:
        if len(cnt) < 5:
            continue

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < 0.82:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        if circles is not None:
            is_circle = False
            for c in circles[0]:
                if np.hypot(cx - c[0], cy - c[1]) < c[2] * 1.1:
                    is_circle = True
                    break
            if is_circle:
                continue

        ellipse = cv2.fitEllipse(cnt)
        (ex, ey), (MA, ma), angle = ellipse
        if ma == 0 or (MA / ma) > 3.0:
            continue

        ellipse_area = np.pi * (MA / 2) * (ma / 2)
        if ellipse_area == 0:
            continue

        if 0.80 < (area / ellipse_area) < 1.20:
            valid_ellipses.append(ellipse)

    return valid_ellipses


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def gambar_garis(img_asli, lines):
    img_hasil = img_asli.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_hasil, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img_hasil


def gambar_lingkaran(img_asli, circles):
    img_hasil = img_asli.copy()
    if circles is not None:
        for i in circles[0, :]:
            cv2.circle(img_hasil, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img_hasil, (i[0], i[1]), 2, (0, 0, 255), 3)
    return img_hasil


def gambar_elips(img_asli, ellipses):
    img_hasil = img_asli.copy()
    for ellipse in ellipses:
        cv2.ellipse(img_hasil, ellipse, (255, 0, 0), 2)
    return img_hasil


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    file_gambar = "photoTest.jpg"

    img, img_gray, img_mask, img_blur_biner, img_blur_gray, img_canny = proses_awal(file_gambar)

    if img is not None:
        print(f"Mengeksekusi analisis pada {file_gambar}...")

        circles  = deteksi_lingkaran(img_blur_gray)
        lines    = deteksi_garis(img_mask)
        ellipses = deteksi_elips(img, img_mask, circles)

        print(f"Garis terdeteksi    : {len(lines) if lines is not None else 0}")
        print(f"Lingkaran terdeteksi: {len(circles[0]) if circles is not None else 0}")
        print(f"Elips terdeteksi    : {len(ellipses)}")

        img_garis     = gambar_garis(img, lines)
        img_lingkaran = gambar_lingkaran(img, circles)
        img_elips     = gambar_elips(img, ellipses)

        # Alur visualisasi yang konsisten:
        # Asli → Grayscale → HSV Mask (biner) → Blur Biner → Canny → hasil deteksi
        cv2.imshow('1. Gambar Asli',               img)
        cv2.imshow('2. Grayscale',                 img_gray)
        cv2.imshow('3. HSV Saturation Mask (Biner)', img_mask)
        cv2.imshow('4. Gaussian Blur (dari Biner)', img_blur_biner)
        cv2.imshow('5. Canny Edge Detection',       img_canny)
        cv2.imshow('6. Deteksi Garis',              img_garis)
        cv2.imshow('7. Deteksi Lingkaran (Hough)',  img_lingkaran)
        cv2.imshow('8. Deteksi Elips (fitEllipse)', img_elips)

        print("Selesai. Tekan tombol apapun pada jendela gambar untuk menutup.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()