import cv2
import numpy as np
import matplotlib.pyplot as plt

net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", r"yolov3_custom_6000.weights")
classes = ['Elon_Musk','Pen','Jeff_Bezos']
img = cv2.imread('sample_images/jeff.jpeg')
img = cv2.resize(img, (1280, 720))
hight, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
output_layers_name = net.getUnconnectedOutLayersNames()
layerOutputs = net.forward(output_layers_name)
boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.7:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * hight)
            w = int(detection[2] * width)
            h = int(detection[3] * hight)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, .5, .4)

boxes = []
confidences = []
class_ids = []

for output in layerOutputs:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * hight)
            w = int(detection[2] * width)
            h = int(detection[3] * hight)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, .8, .4)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))
if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]

        # Center
        xc = x + w//2
        yc = y + h//2

        center_coordinates = (xc, yc)
        # Radius of circle
        radius = 2
        # Blue color in BGR
        color2 = (133,12, 87)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        cv2.circle(img, center_coordinates, radius, color2, thickness)

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + confidence, (x, y + 400), font, 2, color, 2)

cv2.imshow('img', img)
cv2.waitKey()

# cap.release()
cv2.destroyAllWindows()