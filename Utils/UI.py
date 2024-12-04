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
from PIL import Image, ImageDraw, ImageFont


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

imgBackground = cv2.imread(
    "Resources/background.png"
)

folderModePath = "Resources/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

cap = cv2.VideoCapture(0)
cap.set(3, 960)  
cap.set(4, 720)
cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Face Recognition", cv2.WINDOW_FULLSCREEN)
cv2.resizeWindow("Face Recognition", 1920, 1080)

THRESHOLD = 0.7
PRINT_DELAY = 2

detected_name = None
detected_time = None
modeType = 0
counter = 0
predicted_name = "Unknown"

def add_text_with_custom_font(image, text, position, font_size, text_color=(255, 255, 255)):
    font_path = "Font\SFMedium.OTF"
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print("Font tidak ditemukan, menggunakan font default.")
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=text_color)
    image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    return image_with_text


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame_with_background = imgBackground.copy()

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(img_rgb)

    resized_frame = cv2.resize(frame, (960, 720))
    frame_with_background[242 : 242 + 720, 80 : 80 + 960] = resized_frame
    frame_with_background[65 : 65 + 950, 1210 : 1210 + 621] = imgModeList[modeType]

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

                bbox = (160 + x_min, 320 + y_min, 180 + x_max, 330 + y_max)
                cv2.rectangle(
                    frame_with_background,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 255, 0),
                    2,
                )

                current_time = time.time()
                if detected_name == predicted_name and predicted_name != "Unknown":
                    frame_with_background[65 : 65 + 950, 1210 : 1210 + 621] = imgModeList[2]
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
                            imgStudent_resized = cv2.resize(imgStudent, (370, 370))
                            frame_with_background[197 : 197 + 370, 1335 : 1335 + 370] = imgStudent_resized
                            
                            datetimeObject = datetime.strptime(
                                studentInfo["last_attendance_time"], "%Y-%m-%d %H:%M:%S"
                            )
                            
                            secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                            
                            if secondsElapsed > 3600:
                                print(secondsElapsed)
                                ref = db.reference(f"Students/{predicted_name}")
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
                                frame_with_background[65 : 65 + 950, 1210 : 1210 + 621] = imgModeList[modeType]

                        if modeType != 3:
                            if 10 < counter < 20:
                                modeType = 2

                            frame_with_background[65 : 65 + 950, 1210 : 1210 + 621] = imgModeList[modeType]
                            if counter <= 10:
                                frame_with_background = add_text_with_custom_font(
                                    frame_with_background,
                                    "2702303715",
                                    (1430, 643),
                                    35,  
                                    (65, 65, 65)  
                                )
                                frame_with_background = add_text_with_custom_font(
                                    frame_with_background,
                                    "Aldy Revigustian",
                                    (1430, 767),
                                    35,  
                                    (65, 65, 65)  
                                )
                                
                                frame_with_background = add_text_with_custom_font(
                                    frame_with_background,
                                    "LG01",
                                    (1430, 893),
                                    35,  
                                    (65, 65, 65)  
                                )

                                frame_with_background[197 : 197 + 370, 1335 : 1335 + 370] = imgStudent_resized

                            counter += 1
                            if counter >= 5:
                                counter = 0
                                modeType = 0
                                studentInfo = []
                                imgStudent = []
                                frame_with_background[65 : 65 + 950, 1210 : 1210 + 621] = imgModeList[modeType]

                        elif modeType == 3:
                            if 10 < counter < 20:
                                modeType = 3

                            frame_with_background[65 : 65 + 950, 1210 : 1210 + 621] = imgModeList[modeType]
                            if counter <= 10:
                                frame_with_background = add_text_with_custom_font(
                                    frame_with_background,
                                    "2702303715",
                                    (1430, 643),
                                    35,  
                                    (65, 65, 65)  
                                )
                                frame_with_background = add_text_with_custom_font(
                                    frame_with_background,
                                    "Aldy Revigustian",
                                    (1430, 767),
                                    35,  
                                    (65, 65, 65)  
                                )
                                
                                frame_with_background = add_text_with_custom_font(
                                    frame_with_background,
                                    "LG01",
                                    (1430, 893),
                                    35,  
                                    (65, 65, 65)  
                                )

                                frame_with_background[197 : 197 + 370, 1335 : 1335 + 370] = imgStudent_resized

                            counter += 1
                            if counter >= 5:
                                counter = 0
                                modeType = 0
                                studentInfo = []
                                imgStudent = []
                                frame_with_background[65 : 65 + 950, 1210 : 1210 + 621] = imgModeList[modeType]
                else:
                    detected_name = predicted_name
                    detected_time = current_time

    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Recognition", frame_with_background)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
