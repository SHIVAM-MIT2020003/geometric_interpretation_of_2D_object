
from keras.models import load_model
from image_helper_modules import load_image_pixels
from localization import correct_yolo_boxes, decode_netout, do_nms, get_boxes, draw_boxes
from data_set_info import labels


photo_filename = 'images/items.jpg'

model = load_model('models/model.h5')

input_w, input_h = 416, 416

image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

yhat = model.predict(image)


# define the anchors
anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]

# define the probability threshold for detected objects
class_threshold = 0.6

boxes = list()
for i in range(len(yhat)):
    # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
# correct the sizes of the bounding boxes for the shape of the image
correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# suppress non-maximal boxes
do_nms(boxes, 0.5)
# define the labels

# get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

# summarize what we found
# for i in range(len(v_boxes)):
#     print(v_labels[i], v_scores[i])
# draw what we found

for i in range(len(v_boxes)):
    print(v_boxes[i])

draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
