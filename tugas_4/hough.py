import os
import cv2
import numpy as np
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

image_files = [f for f in os.listdir(script_dir) if f.lower().endswith(valid_extensions)]

if not image_files:
    print("Tidak ada file gambar ditemukan di direktori ini.")
    sys.exit()

print("=== DAFTAR GAMBAR ===")
for i, file_name in enumerate(image_files):
    print(f"{i + 1}. {file_name}")

try:
    img_choice = int(input("Pilih nomor gambar yang ingin diproses: ")) - 1
    if img_choice < 0 or img_choice >= len(image_files):
        print("Pilihan tidak valid.")
        sys.exit()
except ValueError:
    print("Input harus berupa angka.")
    sys.exit()

image_file = os.path.join(script_dir, image_files[img_choice])
img = cv2.imread(image_file)

if img is None:
    print("Gagal membaca gambar.")
    sys.exit()

print("\n=== PILIH JENIS DETEKSI ===")
print("1. Hough Transform - Garis")
print("2. Hough Transform - Lingkaran")
print("3. Fit Ellipse - Elips")

try:
    det_choice = int(input("Masukkan pilihan (1/2/3): "))
except ValueError:
    print("Input harus berupa angka.")
    sys.exit()

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
img_result = img.copy()
window_title = ""

edges = None 
img_lines_only = np.zeros_like(img)

if det_choice == 1:
    edges = cv2.Canny(img_blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=500)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(img_lines_only, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    window_title = 'Hasil - Deteksi Garis (Hough)'

elif det_choice == 2:
    circles = cv2.HoughCircles(
        img_blur, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=60,
        param1=100,
        param2=60,
        minRadius=20,
        maxRadius=150
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img_result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(img_result, (i[0], i[1]), 2, (0, 0, 255), 3)
    window_title = 'Hasil - Deteksi Lingkaran (Hough)'

elif det_choice == 3:
    edges = cv2.Canny(img_blur, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        if len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(img_result, ellipse, (255, 0, 0), 2)
    window_title = 'Hasil - Deteksi Elips (fitEllipse)'

else:
    print("Pilihan deteksi tidak valid. Silakan jalankan ulang script.")
    sys.exit()

cv2.imshow('Gambar Asli', img)
cv2.imshow(window_title, img_result)
cv2.imshow('Grayscale', img_gray)
cv2.imshow('Blurred', img_blur)

if edges is not None:
    cv2.imshow('Canny Edges', edges)

if det_choice == 1:
    cv2.imshow('Hanya Garis (Lines)', img_lines_only)

cv2.waitKey(0)
cv2.destroyAllWindows()