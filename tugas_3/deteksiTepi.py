from PIL import Image
import math

def konvolusi_tepi(path_gambar, output_gambar, metode="sobel"):
    img = Image.open(path_gambar).convert("RGB")
    lebar, tinggi = img.size
    img_baru = Image.new("L", (lebar, tinggi))
    
    pixels_lama = img.load()
    pixels_baru = img_baru.load()

    gray_matrix = [[0] * lebar for _ in range(tinggi)]
    for y in range(tinggi):
        for x in range(lebar):
            r, g, b = pixels_lama[x, y]
            gray_matrix[y][x] = int(0.299*r + 0.587*g + 0.114*b)

    if metode == "sobel":
        Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        Ky = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    else: # Prewitt
        Kx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        Ky = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]

    print(f"Deteksi Tepi ({metode.capitalize()})")
    
    # 3. Operasi Konvolusi
    # Kita mulai dari indeks 1 hingga panjang-1 untuk menghindari error di ujung/tepi matriks (border)
    for y in range(1, tinggi - 1):
        for x in range(1, lebar - 1):
            gx = 0
            gy = 0
            
            # Perkalian matriks kernel 3x3
            for ky in range(-1, 2):
                for kx in range(-1, 2):
                    pixel_val = gray_matrix[y + ky][x + kx]
                    gx += pixel_val * Kx[ky + 1][kx + 1]
                    gy += pixel_val * Ky[ky + 1][kx + 1]
            
            # Rumus Magnitude (Teorema Pythagoras untuk mendapatkan tepi akhir)
            magnitude = int(math.sqrt(gx**2 + gy**2))
            
            pixels_baru[x, y] = max(0, min(255, magnitude))
            
    img_baru.save(output_gambar)
    print(f"{output_gambar}")

if __name__ == "__main__":
    konvolusi_tepi("gambar.jpg", "tepiSobel.jpg", metode="sobel")
    konvolusi_tepi("gambar.jpg", "tepiPrewitt.jpg", metode="prewitt")
