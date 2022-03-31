from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg

import matplotlib.pyplot as plt
import random
import cv2


def get_train_config(config_file, checkpoint, train_dataset, test_dataset, num_classes, device, output_path):
    config = get_cfg()
    config.merge_from_file(model_zoo.get_config_file(config_file))
    config.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint)
    config.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    config.MODEL.DEVICE = device
    config.DATASETS.TRAIN = (train_dataset,)
    config.DATASETS.TEST = (test_dataset,)
    config.DATALOADER.NUM_WORKERS = 2
    config.SOLVER.IMS_PER_BATCH = 2
    config.SOLVER.BASE_LR = 0.0001
    config.SOLVER.MAX_ITER = 3000
    config.SOLVER.STEPS = []
    config.OUTPUT_DIR = output_path

    return config


def vis_image(image_path, predictor):
    print(image_path)
    image = cv2.imread(image_path)

    outputs = predictor(image)
    visualizer = Visualizer(image[:, :, ::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
    visualizer = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(visualizer.get_image())
    plt.show()

