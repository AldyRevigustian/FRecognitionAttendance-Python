import os
from facenet_pytorch import MTCNN
from PIL import Image

mtcnn = MTCNN(keep_all=True)

dataset_path = "Augmented_Images"
def delete_images_without_faces(dataset_path):
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)

                try:
                    img = Image.open(img_path).convert("RGB")
                    faces, _ = mtcnn(img, return_prob=True)
                    if faces is None or len(faces) == 0:
                        print(f"Deleting {img_path} - No face detected.")
                        os.remove(img_path)  # Hapus gambar

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

delete_images_without_faces(dataset_path)
