from detectron2.engine import DefaultPredictor
import os
import pickle
import glob
from utils import *

config_save_path = r"C:\Users\anton\Documents\DetectronDataset/cfg.pickle"
with open(config_save_path, 'rb') as conf:
    config = pickle.load(conf)
output_path = r"C:\Users\anton\Documents\DetectronDataset\output"
config.MODEL.WEIGHTS = os.path.join(output_path, "model_final.pth")
config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
config.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(config)

image_path = r"C:\Users\anton\Documents\DetectronDataset\evaluation"
eval = []
for foundImage in glob.glob(f'{image_path}/*.png'):
    eval.append(os.path.basename(foundImage))
    vis_image(foundImage, predictor)