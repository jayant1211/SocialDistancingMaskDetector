from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import os
import cv2
from math import pow, sqrt
#import playsound

# detecting and predicting mask in a frame
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))

    # passing blob to face detection model
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, locations and  predictions array
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extracting the confidence(probability)
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            #face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        print('no of fcaes:',len(faces))
        # for faster inference we'll make batch predictions on *all* faces at the same time rather than one-by-one predictions in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding locations
    return (locs, preds)


# Parse the arguments from command line

labels = [line.strip() for line in open('ssd\class_labels.txt')]
#print(labels)

# Generate random bounding box bounding_box_color for each label
bounding_box_color = np.random.uniform(0, 255, size=(len(labels), 3))

print("[INFO] loading face detector model...")

prototxtPath = os.path.sep.join(['face_detector', "deploy.prototxt"])
weightsPath = os.path.sep.join(['face_detector', "res10_300x300_ssd_iter_140000.caffemodel"])

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#mask detector model
print("[INFO] loading face mask detector model...")
maskNet = load_model('model/mask_detector.model')

# Load model
print("\nLoading model...\n")
network = cv2.dnn.readNetFromCaffe('SSD/SSD_MobileNet_prototxt.txt', 'SSD/SSD_MobileNet.caffemodel')

print("\nStreaming video using device...\n")
# Capture video from file or through device

cap = cv2.VideoCapture('final.mp4')
#frame = cv2.imread('2.jpg')
frame_no = 0

vdo = cv2.VideoWriter('social.avi', cv2.VideoWriter_fourcc(*'XVID'),10, (800,500))

while True:

    frame_no = frame_no + 1
    # Capture one frame after another

    ret, frame = cap.read()

    if ret == True:
        frame = cv2.flip(frame, 1)
        # frame = cv2.resize(frame,(400,400))

        (h, w) = frame.shape[:2]
        vdo.write(frame)

        # Resize the frame to suite the model requirements. Resize the frame to 300X300 pixels
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

        network.setInput(blob)
        detections = network.forward()

        pos_dict = dict()
        coordinates = dict()

        # Focal length
        F = 500

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            # print(confidence)
            if confidence > 0.3:

                class_id = int(detections[0, 0, i, 1])

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype('int')

                # Filtering only persons class. Class Id of 'person' is 15
                if class_id == 15.00:

                    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

                    # loop over the detected face locations and their corresponding locations
                    for (box, pred) in zip(locs, preds):
                        # unpack the bounding box and predictions
                        (startX_mask, startY_mask, endX_mask, endY_mask) = box
                        (mask, withoutMask) = pred

                        # determine the class label and color we'll use to draw
                        # the bounding box and text
                        if mask > 0.7:
                            label_mask = "Mask"
                            color = (0, 255, 0)
                            # ALARM_ON = False
                            # sound_alarm()
                        if withoutMask > 0.7:
                            label_mask = "No Mask"
                            color = (0, 0, 255)
                            # if not ALARM_ON:
                            #   ALARM_ON = True
                            # sound_alarm()

                        # Draw bounding box for the object
                        cv2.rectangle(frame, (startX, startY), (endX, endY), bounding_box_color[class_id], 4)

                        label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                        label_mask = "{}: {:.2f}%".format(label_mask, max(mask, withoutMask) * 100)
                        print("{}".format(label))
                        # cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                        cv2.putText(frame, label_mask, (startX_mask, startY_mask - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    color, 2)
                        cv2.rectangle(frame, (startX_mask, startY_mask), (endX_mask, endY_mask), color, 4)

                    coordinates[i] = (startX, startY, endX, endY)

                    # Mid point of bounding box
                    x_mid = round((startX + endX) / 2, 4)
                    y_mid = round((startY + endY) / 2, 4)

                    width = round(endX - startX, 4)

                    # Distance from camera based on triangle similarity
                    distance = (40 * F) / width  # (F = (P x D) / W)
                    print("Distance(cm):{dist}\n".format(dist=distance))

                    # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                    x_mid_cm = (x_mid * distance) / F
                    y_mid_cm = (y_mid * distance) / F
                    pos_dict[i] = (x_mid_cm, y_mid_cm, distance)

        # Distance between every object detected in a frame
        close_objects = set()
        for i in pos_dict.keys():
            for j in pos_dict.keys():
                if i < j:
                    dist = sqrt(pow(pos_dict[i][0] - pos_dict[j][0], 2) + pow(pos_dict[i][1] - pos_dict[j][1], 2) + pow(
                        pos_dict[i][2] - pos_dict[j][2], 2))

                    # Check if distance less than 2 metres or 200 centimetres
                    if dist < 50:
                        close_objects.add(i)
                        close_objects.add(j)

        for i in pos_dict.keys():
            if i in close_objects:
                COLOR = (0, 0, 255)
            else:
                COLOR = (0, 255, 0)
                # ALARM_ON = False

            if COLOR == (0, 0, 255):
                print("threat")
                # sound_alarm()
            (startX, startY, endX, endY) = coordinates[i]

            cv2.rectangle(frame, (startX, startY), (endX, endY), COLOR, 3)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # Convert cms to feet
            cv2.putText(frame, 'Person: {i} ft'.format(i=round(pos_dict[i][2] / 30.48, 4)), (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)

        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

        # Show frame
        frame = cv2.resize(frame, (800, 500))
        cv2.imshow('Frame', frame)

        vdo.write(frame)
        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break
        # cv2.waitKey()
    else:
        break


# Clean
vdo.release()
cap.release()
