"""
Calibration Tool - Klik pada warna pita resistor untuk merekam nilai LAB-nya.
Jalankan ini sekali per warna, lalu copy hasilnya ke resistor_detector.py

Cara pakai:
  python calibrate.py resistor1.png

- Klik pada piksel warna yang ingin direkam
- Tekan 's' untuk menyimpan warna terakhir yang diklik
- Tekan 'q' untuk keluar dan melihat semua hasil
"""

import cv2
import numpy as np
import sys

recorded = []
last_click = None
img_display = None

def on_mouse(event, x, y, flags, param):
    global last_click, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param['img']
        bgr = img[y, x].tolist()
        lab_px = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0].tolist()
        last_click = {'x': x, 'y': y, 'bgr': bgr, 'lab': lab_px}

        # Draw crosshair
        disp = img.copy()
        cv2.circle(disp, (x, y), 8, (0, 255, 0), 2)
        info = f"BGR={bgr}  LAB={lab_px}"
        cv2.putText(disp, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3)
        cv2.putText(disp, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)
        cv2.putText(disp, "Tekan 's' simpan, klik warna lain, 'q' keluar", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.imshow("Calibration", disp)
        print(f"Diklik: BGR={bgr}  LAB={lab_px}")

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "resistor3.jpg"
    img_orig = cv2.imread(path)
    if img_orig is None:
        print(f"Error: tidak bisa buka '{path}'")
        return

    img = cv2.resize(img_orig, (900, 550))
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse, {'img': img})

    disp = img.copy()
    cv2.putText(disp, "Klik warna pita -> 's' simpan -> 'q' selesai", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3)
    cv2.putText(disp, "Klik warna pita -> 's' simpan -> 'q' selesai", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,200,255), 1)
    cv2.imshow("Calibration", disp)

    color_names = [
        'Hitam','Cokelat','Merah','Oranye','Kuning',
        'Hijau','Biru','Ungu','Abu-abu','Putih','Emas','Perak'
    ]
    idx = 0
    print(f"\nKlik pada warna: {color_names[idx]}")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s') and last_click:
            name = color_names[idx] if idx < len(color_names) else f"Warna{idx}"
            entry = {'name': name, **last_click}
            recorded.append(entry)
            print(f"  ✓ Disimpan: {name} -> BGR={last_click['bgr']}")
            idx += 1
            if idx < len(color_names):
                print(f"\nKlik pada warna: {color_names[idx]}")
            else:
                print("\nSemua warna sudah direkam! Tekan 'q' untuk lihat hasil.")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("HASIL KALIBRASI - Copy ke resistor_detector.py:")
    print("="*60)
    print("COLOR_BGR_REFS = {")
    for r in recorded:
        b, g, rv = r['bgr']
        print(f"    '{r['name']}':   ({b}, {g}, {rv}),")
    print("}")

if __name__ == "__main__":
    main()