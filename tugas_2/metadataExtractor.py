from PIL import Image

def analisis_citra(path_gambar):
    # Membaca gambar dari storage ke memori
    img = Image.open(path_gambar)
    
    # 1. Ekstraksi Properti Citra
    lebar, tinggi = img.size
    format_img = img.format
    mode_img = img.mode
    
    print("=== PROPERTI CITRA ===")
    print(f"Format File : {format_img}")
    print(f"Resolusi    : {lebar} x {tinggi} piksel")
    print(f"Mode Warna  : {mode_img}")
    
    # 2. Menentukan Depth Color (Kedalaman Warna)
    # Sebuah citra RGB memiliki 3 channel (Red, Green, Blue).
    # Masing-masing channel biasanya dialokasikan 8 bit (total 24 bit).
    if mode_img == 'RGB':
        depth_color = 24
    elif mode_img == 'RGBA':
        depth_color = 32 # Tambahan 8 bit untuk channel Alpha (transparansi)
    elif mode_img == 'L':
        depth_color = 8  # Grayscale (hitam putih)
    else:
        depth_color = "Lainnya"
        
    print(f"Depth Color : {depth_color}-bit\n")
    
    # 3. Ekstraksi Bits/Warna di Semua Titik
    # Kita akan melakukan iterasi (looping) ke setiap koordinat piksel (X, Y)
    # dan menyimpannya ke file teks, karena jika di-print ke terminal akan terlalu panjang.
    pixels = img.load()
    nama_file_output = "matriks_warna.txt"
    
    print("Mengekstrak nilai RGB pada semua titik piksel...")
    with open(nama_file_output, 'w') as f:
        # Loop baris (Y) dari atas ke bawah
        for y in range(tinggi):
            baris = []
            # Loop kolom (X) dari kiri ke kanan
            for x in range(lebar):
                # pixel_value akan berisi tuple, e.g., (255, 120, 0)
                pixel_value = pixels[x, y] 
                baris.append(str(pixel_value))
            
            # Menulis satu baris penuh ke file teks
            f.write(" | ".join(baris) + "\n")
            
    print(f"Selesai! Matriks warna telah diekspor ke '{nama_file_output}'")

if __name__ == "__main__":
    # Pastikan file 'gambar.jpg' berada di folder yang sama dengan script ini
    analisis_citra("gambar.jpg")