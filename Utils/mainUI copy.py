import cv2
import torch
import numpy as np
import joblib
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://faceattendance-b3954-default-rtdb.asia-southeast1.firebasedatabase.app/"
    },
)

device = "cuda" if torch.cuda.is_available() else "cpu"

inception_resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
svm_model = joblib.load("svm_model.pkl")
label_encoder_classes = np.load("label_encoder_classes.npy")
known_embeddings = np.load("known_face_embeddings.npy")
mtcnn = MTCNN(keep_all=True, device=device)

# Load resources
imgBackground = cv2.imread(
    "Resources/background.png"
)  # Load background image only once
folderModePath = "Resources/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition", 1080, 720)

# Constants
THRESHOLD = 0.7
PRINT_DELAY = 2

# State variables
detected_name = None
detected_time = None
modeType = 0
counter = 0
predicted_name = "Unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame_with_background = imgBackground.copy()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)

    frame_with_background[162 : 162 + 480, 55 : 55 + 640] = frame
    frame_with_background[44 : 44 + 633, 808 : 808 + 414] = imgModeList[modeType]

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(frame.shape[1], x_max)
            y_max = min(frame.shape[0], y_max)

            face = img_rgb[y_min:y_max, x_min:x_max]
            if face.size != 0:
                face_resized = cv2.resize(face, (160, 160))
                face_tensor = (
                    torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
                ).to(device)
                face_embedding = (
                    inception_resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy()
                )

                predicted_label = svm_model.predict(face_embedding)[0]
                predicted_name = label_encoder_classes[predicted_label]

                similarity_scores = cosine_similarity(face_embedding, known_embeddings)
                if np.max(similarity_scores) < THRESHOLD:
                    predicted_name = "Unknown"

                bbox = (55 + x_min, 162 + y_min, 55 + x_max, 162 + y_max)
                cv2.rectangle(
                    frame_with_background,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_with_background,
                    predicted_name,
                    (55 + x_min, 162 + y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                current_time = time.time()
                if detected_name == predicted_name and predicted_name != "Unknown":
                    if current_time - detected_time >= PRINT_DELAY:
                        print(f"Name detected: {predicted_name}")
                        if counter == 0:
                            counter = 1
                            modeType = 1
                        detected_time = current_time
                    if counter != 0:
                        if counter == 1:
                            studentInfo = db.reference(f"Students/{predicted_name}").get()
                            print(studentInfo)

                            imgPath = f"Images/{predicted_name}/0.jpg"
                            if not os.path.exists(imgPath):
                                print(f"Error: No such file '{imgPath}' in local storage.")
                                counter = 0
                                modeType = 0
                                continue

                            imgStudent = cv2.imread(imgPath)
                            imgStudent_resized = cv2.resize(imgStudent, (216, 216))
                            frame_with_background[175 : 175 + 216, 909 : 909 + 216] = imgStudent_resized
                            datetimeObject = datetime.strptime(
                                studentInfo["last_attendance_time"], "%Y-%m-%d %H:%M:%S"
                            )
                            secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                            if secondsElapsed > 3600:
                                ref = db.reference(f"Students/{id}")
                                studentInfo["total_attendance"] += 1
                                studentInfo["attendance"] = "P"
                                ref.child("total_attendance").set(
                                    studentInfo["total_attendance"]
                                )
                                ref.child("last_attendance_time").set(
                                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                )
                            else:
                                modeType = 3
                                counter = 0
                                frame_with_background[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                    modeType
                                ]

                        if modeType != 3:
                            if 10 < counter < 20:
                                modeType = 2

                            frame_with_background[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                modeType
                            ]

                            if counter <= 10:
                                cv2.putText(
                                    frame_with_background,
                                    str(studentInfo["total_attendance"]),
                                    (1080, 630),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    1,
                                    (255, 255, 255),
                                    1,
                                )
                                cv2.putText(
                                    frame_with_background,
                                    str(studentInfo["branch"]),
                                    (950, 550),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.4,
                                    (255, 255, 255),
                                    1,
                                )
                                cv2.putText(
                                    frame_with_background,
                                    str(id),
                                    (950, 493),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    0.5,
                                    (255, 255, 255),
                                    1,
                                )
                                (w, h), _ = cv2.getTextSize(
                                    studentInfo["name"], cv2.FONT_HERSHEY_COMPLEX, 1, 1
                                )
                                offset = (414 - w) // 2
                                cv2.putText(
                                    frame_with_background,
                                    str(studentInfo["name"]),
                                    (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX,
                                    1,
                                    (255, 255, 255),
                                    1,
                                )

                                frame_with_background[175 : 175 + 216, 909 : 909 + 216] = (
                                    imgStudent_resized
                                )

                            counter += 1
                            if counter >= 20:
                                counter = 0
                                modeType = 0
                                studentInfo = []
                                imgStudent = []
                                frame_with_background[44 : 44 + 633, 808 : 808 + 414] = imgModeList[
                                    modeType
                                ]
                else:
                    detected_name = predicted_name
                    detected_time = current_time

    else:
        modeType = 0
        counter = 0

    # Tampilkan hasil dalam jendela yang sama
    cv2.imshow("Face Recognition", frame_with_background)

    # Exit jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
