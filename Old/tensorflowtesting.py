import tensorflow as tf
from imageTrainer import *
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
train_model("E:/Computing Project BSc/AI Files/roadobjects", "stopsign", 8, 5,
            "E:/Computing Project BSc/AI Files/PreTrain/pretrained-yolov3.h5")
