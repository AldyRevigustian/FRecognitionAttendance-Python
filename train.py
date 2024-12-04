import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from PIL import Image
import joblib

device = "cpu"
# device = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=True, device=device)
inception_resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
dataset_path = "Augmented_Images"


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        for person_name in os.listdir(root_dir):
            person_folder = os.path.join(root_dir, person_name)
            if os.path.isdir(person_folder):
                for img_name in os.listdir(person_folder):
                    img_path = os.path.join(person_folder, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(person_name)

        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
            return None, None

        if self.transform:
            img = self.transform(img)

        faces, probs = mtcnn(img, return_prob=True)

        if faces is not None and len(faces) > 0:
            face = faces[0]
            return face, label
        else:
            print(f"No face detected in image {img_path}")
            return None, None


dataset = FaceDataset(dataset_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

face_embeddings = []
labels = []
failed_images = []
known_face_embeddings = []

for faces, label in dataloader:
    if faces is not None:
        try:
            embeddings = inception_resnet(faces.to(device))
            face_embeddings.append(embeddings.detach().cpu().numpy())
            known_face_embeddings.append(
                inception_resnet(faces[0].unsqueeze(0).to(device))
                .detach()
                .cpu()
                .numpy()
            )
            labels.append(label.numpy())
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
    else:
        failed_images.append(label)

if len(face_embeddings) == 0 or len(labels) == 0:
    print(
        "Tidak ada data yang dapat digunakan untuk pelatihan. Cek kembali dataset Anda."
    )
else:
    known_face_embeddings = np.vstack(known_face_embeddings)
    face_embeddings = np.vstack(face_embeddings) if face_embeddings else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])

    svm_model = SVC(kernel="linear", probability=True)
    svm_model.fit(face_embeddings, labels)

    np.save("Model/label_encoder_classes.npy", dataset.label_encoder.classes_)
    np.save("Model/known_face_embeddings.npy", known_face_embeddings)
    joblib.dump(svm_model, "Model/svm_model.pkl")

    print("Model telah dilatih dan disimpan.")
