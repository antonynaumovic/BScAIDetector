from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *

config_save_path = "/home/ant/Documents/Detectron2/Dataset/cfg.pickle"
with open(config_save_path, 'rb') as conf:
    config = pickle.load(conf)
output_path = r"/home/ant/Documents/Detectron2/Python/output/"
config.MODEL.WEIGHTS = os.path.join(output_path, "model_final.pth")
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(config)

#image_path = "/home/ant/Documents/Detectron2/Dataset/testing/Image0243.png"
image_path = "/home/ant/Documents/Detectron2/Dataset/testing/eval1.jpeg"

vis_image(image_path, predictor)