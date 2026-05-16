from PIL import Image

def skalalisasi(path_gambar, output_gambar, skala):
    img = Image.open(path_gambar).convert("RGB")
    lebar_lama, tinggi_lama = img.size
    
    # Menghitung resolusi baru
    lebar_baru = int(lebar_lama * skala)
    tinggi_baru = int(tinggi_lama * skala)
    
    img_baru = Image.new("RGB", (lebar_baru, tinggi_baru))
    
    pixels_lama = img.load()
    pixels_baru = img_baru.load()

    print(f"(Skala {skala}x)...")
    
    rasio_x = lebar_lama / lebar_baru
    rasio_y = tinggi_lama / tinggi_baru
    
    for y_baru in range(tinggi_baru):
        for x_baru in range(lebar_baru):
            # Rumus Nearest Neighbor: Mencari koordinat asli dari piksel baru
            x_lama = int(x_baru * rasio_x)
            y_lama = int(y_baru * rasio_y)
            
            # Menyalin piksel dari koordinat asli ke kanvas baru
            pixels_baru[x_baru, y_baru] = pixels_lama[x_lama, y_lama]
            
    img_baru.save(output_gambar)
    print(f"{output_gambar} (Resolusi: {lebar_baru}x{tinggi_baru})")

if __name__ == "__main__":
    skalalisasi("gambar.jpg", "hasilSkala.jpg", 2.0)
    # Anda juga bisa memperbesar dengan skala > 1 (misal 2.0)
