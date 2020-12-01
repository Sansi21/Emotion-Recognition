from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/best_model.hdf5' 

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False) 
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]

# print(emotion_classifier.summary())


# starting video streaming
cv2.namedWindow('Display Screen')
camera = cv2.VideoCapture(0) 
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    #tries to find 5 faces in 1 frame
    canvas = np.zeros((600, 600, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0] #sabse zyada area wala face lelia
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW] #face nikal lia
        roi = cv2.resize(roi, (64, 64)) 
        roi = roi.astype("float") / 255.0 #normalizing data 0-1
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0) 
        
        
        preds = emotion_classifier.predict(roi)[0] #array of array = 1 array [[1,2,3....7]]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()] #index of maximum prob
    else: continue

 #zip - ek array bana dia
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)): 
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]
                
                w = int(prob * 500)
                cv2.rectangle(canvas, (7, (i * 45) + 5),(w, (i * 45) + 45), (200, 20, 0), -1)
                cv2.putText(canvas, text, (10, (i * 45) + 33),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1,(255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.9, (255, 255, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),(20, 200, 0), 2)
                
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


    cv2.imshow('Display_screen', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()