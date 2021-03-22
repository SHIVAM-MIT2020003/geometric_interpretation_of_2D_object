from yolo_modules import _conv_block, make_yolov3_model
from weight_reader import WeightReader


model = make_yolov3_model()
#training...

weight_reader = WeightReader('transfer_learning_weights/yolov3.weights')
weight_reader.load_weights(model)
model.save('models/model.h5')
