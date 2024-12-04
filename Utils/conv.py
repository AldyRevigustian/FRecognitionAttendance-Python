from PIL import Image
import os

def convert_png_to_jpg(input_dir, output_dir):
    # Menelusuri semua folder dan file di direktori input
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".png"):
                # Menentukan path file PNG
                png_path = os.path.join(root, filename)

                # Membuka file PNG
                img = Image.open(png_path)

                # Mengonversi PNG ke RGB (karena JPG tidak mendukung transparansi)
                rgb_img = img.convert("RGB")

                # Menyimpan file sebagai JPG dengan struktur folder yang sama
                relative_path = os.path.relpath(root, input_dir)  # Mendapatkan subfolder relatif
                jpg_folder = os.path.join(output_dir, relative_path)  # Folder tujuan di output

                # Membuat folder jika belum ada
                if not os.path.exists(jpg_folder):
                    os.makedirs(jpg_folder)

                jpg_filename = f"{os.path.splitext(filename)[0]}.jpg"
                jpg_path = os.path.join(jpg_folder, jpg_filename)

                # Menyimpan gambar JPG
                rgb_img.save(jpg_path)
                print(f"Berhasil mengonversi {filename} menjadi {jpg_filename} di {jpg_folder}")

# Direktori input dan output
input_directory = "scraped_images/"
output_directory = "Images/"

# Panggil fungsi untuk mengonversi PNG ke JPG
convert_png_to_jpg(input_directory, output_directory)
