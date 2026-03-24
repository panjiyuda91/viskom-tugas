from PIL import Image

def ubah_contrast(path_gambar, output_gambar, level_kontras):
    img = Image.open(path_gambar).convert("RGB")
    lebar, tinggi = img.size
    img_baru = Image.new("RGB", (lebar, tinggi))
    
    pixels_lama = img.load()
    pixels_baru = img_baru.load()

    # Rumus Faktor Kontras standar
    faktor = (259 * (level_kontras + 255)) / (255 * (259 - level_kontras))

    print("Memproses Contrast...")
    for y in range(tinggi):
        for x in range(lebar):
            r, g, b = pixels_lama[x, y]
            
            # Rumus peregangan nilai kontras
            r_baru = int(faktor * (r - 128) + 128)
            g_baru = int(faktor * (g - 128) + 128)
            b_baru = int(faktor * (b - 128) + 128)
            
            # Memastikan nilai tetap dalam rentang 8-bit (0-255)
            r_baru = max(0, min(255, r_baru))
            g_baru = max(0, min(255, g_baru))
            b_baru = max(0, min(255, b_baru))
            
            pixels_baru[x, y] = (r_baru, g_baru, b_baru)
            
    img_baru.save(output_gambar)
    print(f"Selesai! Disimpan sebagai {output_gambar}")

if __name__ == "__main__":
    # Meningkatkan kontras sebesar 50
    ubah_contrast("gambar.jpg", "hasilContrast.jpg", 50)