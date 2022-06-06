#import face_recognition
import numpy as np
import argparse
import cv2
import os
from os import listdir
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
# construct the argument parse and parse the arguments
folder = r"C:/Program Files/projectimg"
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                help="path to input image")
ap.add_argument("-f", "--face", required=True,
                help="path to face detector model directory")
ap.add_argument("-a", "--age", required=True,
                help="path to age detector model directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
               "(38-43)", "(48-53)", "(60-100)"]
print("[INFO] loading face detector model...")
prototxtPath = '/home/dlinano/real-time-age-detection-using-webcam-main/deploy.prototext.txt'
weightsPath = '/home/dlinano/real-time-age-detection-using-webcam-main/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load our serialized age detector model from disk
print("[INFO] loading age detector model...")
prototxtPath = '/home/dlinano/real-time-age-detection-using-webcam-main/age_deploy.prototext.txt'
weightsPath = '/home/dlinano/real-time-age-detection-using-webcam-main/age_net.caffemodel'
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)
#image = cv2.imread(args["image"])
#image = cv2.resize(image, (720, 640))
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
while True:
    ret, image = cap.read()
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)
    for (x, y, w, h) in faces:
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        roi_color = image[y:y + h, x:x + w]
        cv2.imwrite(str(w) + str(h) + '_faces.png', roi_color)
        try:
           # unknown_picture = face_recognition.load_image_file(
           #     'C:/Users/Dell/' + str(w) + str(h) + '_faces.png')
            image=cv2.imread(str(w) + str(h) + '_faces.png')
            image = cv2.cvtColor(unknown_picture, cv2.COLOR_BGR2RGB)
            (h, w) = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                         (104.0, 177.0, 123.0))
            # pass the blob through the network and obtain the face
            # detections
            print("[INFO] computing face detections...")
            faceNet.setInput(blob)
            detections = faceNet.forward()
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > args["confidence"]:
                    # compute the (x, y)-coordinates of the bounding box for the
                    # object
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # extract the ROI of the face and then construct a blob from
                    # *only* the face ROI
                    face = image[startY:endY, startX:endX]
                    faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
                ageNet.setInput(faceBlob)
                preds = ageNet.forward()
                i = preds[0].argmax()
                age = AGE_BUCKETS[i]
                ageConfidence = preds[0][i]
                # display the predicted age to our terminal
                text = "{}: {:.2f}%".format(age, ageConfidence * 100)
                print("[INFO] {}".format(text))
                # draw the bounding box of the face along with the associated
                # predicted age
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(image, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(image, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        except:
            print("image problem")
    cv2.imshow("face_detect", image)
    cv2.waitKey(0)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')
