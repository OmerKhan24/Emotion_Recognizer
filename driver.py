import cv2
from keras.models import model_from_json
import numpy as np

json_file = open("driver_model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("driver_model.h5")

face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_cap = cv2.VideoCapture(0)

if not video_cap.isOpened():
    print("Error: Could not open video device.")
    exit()

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0

labels = {0 : "angry", 1 : "disgust", 2 : "fear", 3 : "happy", 4 : "neutarl", 5 : "sad", 6 : "suprise"}

while True:

    ret, video_data = video_cap.read()
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = col[y : y + h, x : x + w]
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        predict = model.predict(img)
        prediction_label = labels[predict.argmax()]
        cv2.putText(video_data, '% s' %(prediction_label), (w-10, y-10),cv2.FONT_HERSHEY_COMPLEX_SMALL,2, (0,0,255))


    cv2.imshow("Live Video", video_data)
    
    if cv2.waitKey(1) == ord('q'):
        break

