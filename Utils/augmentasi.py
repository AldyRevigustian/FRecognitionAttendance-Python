import os
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
from imgaug import augmenters as iaa

# Direktori dataset dan output
dataset_dir = "Images/"
output_dir = "Augmented_Images/"

# Buat folder output jika belum ada
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Inisialisasi detektor wajah
detector = MTCNN()

# Definisi augmentasi gambar
seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # Horizontal flip
        iaa.Affine(rotate=(-30, 30)),  # Rotasi wajah
        iaa.LinearContrast((0.9, 1.1)),  # Kontras
        iaa.Multiply((0.8, 1.2)),  # Kecerahan
        iaa.Grayscale(alpha=(0.0, 1.0)),  # Grayscale
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # Noise ringan
        iaa.AddToHueAndSaturation((-10, 10)),  # Hue dan Saturasi
        iaa.MultiplyHueAndSaturation((0.8, 1.2)),  # Jitter warna
    ]
)


# Fungsi untuk mendeteksi wajah dan memotongnya
def detect_and_crop_face(image_path):
    img = Image.open(image_path)
    img_rgb = np.array(img)

    boxes, probs = detector.detect(img_rgb)

    if boxes is not None and len(boxes) > 0:
        box = boxes[0]
        x_min, y_min, x_max, y_max = map(int, box)
        img_cropped = img.crop((x_min, y_min, x_max, y_max))
        img_cropped = img_cropped.resize((160, 160))
        return img_cropped
    else:
        return None


# Fungsi untuk augmentasi dan menyimpan gambar
def augment_and_save(face, person_name, base_filename, num_augments=5):
    # Simpan gambar asli terlebih dahulu
    person_folder = os.path.join(output_dir, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    original_filename = os.path.join(person_folder, f"{base_filename}_original.jpg")
    face.save(original_filename)  # Simpan gambar asli

    # Augmentasi dan simpan gambar hasil augmentasi
    for i in range(num_augments):
        augmented_face = seq(image=np.array(face))

        if augmented_face is None or augmented_face.size == 0:
            print(
                f"Warning: Gambar kosong setelah augmentasi untuk {base_filename}_aug_{i+1}"
            )
            continue

        augmented_image = Image.fromarray(augmented_face)
        augmented_image = augmented_image.convert("RGB")

        new_filename = os.path.join(person_folder, f"{base_filename}_aug_{i+1}.jpg")

        try:
            augmented_image.save(new_filename)
        except ValueError as e:
            print(f"Error: Tidak dapat menyimpan gambar {new_filename} - {e}")

        # Setelah menyimpan gambar, kita cek apakah wajah masih terdeteksi
        if not detect_face(new_filename):
            print(
                f"Warning: Wajah tidak terdeteksi pada {new_filename}, menghapus gambar."
            )
            os.remove(new_filename)


# Fungsi untuk mendeteksi wajah pada gambar
def detect_face(image_path):
    img = Image.open(image_path)
    img_rgb = np.array(img)

    boxes, probs = detector.detect(img_rgb)

    if boxes is not None and len(boxes) > 0:
        return True
    else:
        return False


# Fungsi untuk memproses gambar untuk setiap orang
def process_person_images(person_name):
    person_folder = os.path.join(dataset_dir, person_name)
    images = [f for f in os.listdir(person_folder) if f.endswith(".jpg")]

    for image_filename in images:
        image_path = os.path.join(person_folder, image_filename)
        face = detect_and_crop_face(image_path)

        if face is not None:
            base_filename = image_filename.split(".")[0]
            augment_and_save(face, person_name, base_filename)


for person_name in os.listdir(dataset_dir):
    person_folder = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_folder):
        process_person_images(person_name)

print("Augmentasi selesai.")
