from PIL import Image

def ubah_brightness(path_gambar, output_gambar, nilai_brightness):
    img = Image.open(path_gambar).convert("RGB")
    lebar, tinggi = img.size
    # Membuat kanvas kosong baru
    img_baru = Image.new("RGB", (lebar, tinggi))
    
    pixels_lama = img.load()
    pixels_baru = img_baru.load()

    print("Memproses Brightness...")
    for y in range(tinggi):
        for x in range(lebar):
            r, g, b = pixels_lama[x, y]
            
            # Rumus Brightness (penambahan skalar) dengan batasan 0-255
            r_baru = max(0, min(255, r + nilai_brightness))
            g_baru = max(0, min(255, g + nilai_brightness))
            b_baru = max(0, min(255, b + nilai_brightness))
            
            pixels_baru[x, y] = (r_baru, g_baru, b_baru)
            
    img_baru.save(output_gambar)
    print(f"Selesai! Disimpan sebagai {output_gambar}")

if __name__ == "__main__":
    # Menambah kecerahan sebesar 50
    ubah_brightness("gambar.jpg", "hasilBrightness.jpg", 50)