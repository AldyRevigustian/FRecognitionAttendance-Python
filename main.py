import cv2
import torch
import numpy as np
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


device = "cuda" if torch.cuda.is_available() else "cpu"
inception_resnet = InceptionResnetV1(pretrained="vggface2").eval()
svm_model = joblib.load("svm_model.pkl")
label_encoder = LabelEncoder()


label_encoder_classes = np.load("label_encoder_classes.npy")
known_embeddings = np.load(
    "known_face_embeddings.npy"
)  # Pastikan ini file yang berisi embedding wajah yang dikenal

mtcnn = MTCNN(keep_all=True)
cap = cv2.VideoCapture(0)
THRESHOLD = 0.7


scaler = StandardScaler()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(img_rgb)

    if boxes is not None and len(boxes) > 0:
        box = boxes[0]

        if box[2] - box[0] > 20 and box[3] - box[1] > 20:
            face = img_rgb[int(box[1]) : int(box[3]), int(box[0]) : int(box[2])]

            if face.size != 0:
                face_resized = cv2.resize(face, (160, 160))
                face_tensor = (
                    torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
                )
                face_embedding = (
                    inception_resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
                )

                predicted_label = svm_model.predict(face_embedding)[0]
                predicted_name = label_encoder_classes[predicted_label]

                # Hitung cosine similarity dengan wajah yang dikenal
                similarity_scores = cosine_similarity(face_embedding, known_embeddings)

                print(np.max(similarity_scores))
                if np.max(similarity_scores) < THRESHOLD:
                    predicted_name = "Unknown"

                # Gambar kotak wajah
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    predicted_name,
                    (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            else:
                print("Face crop is empty, skipping...")
        else:
            print("Invalid bounding box size, skipping...")

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
