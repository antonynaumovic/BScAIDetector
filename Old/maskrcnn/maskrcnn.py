import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob
import skimage.color
import skimage.io
import skimage.transform
from skimage.filters import threshold_otsu

# Import Mask RCNN
sys.path.append(r"C:\Users\anton\Documents\MaskRcnn\Mask_RCNN")  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


def getWeights(root_path):

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(root_path, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(root_path, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    return COCO_MODEL_PATH


class TrafficConfig(Config):

    # Give the configuration a recognizable name
    NAME = "stopsign"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + stopsign

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    IMAGE_MIN_SCALE = 4

    USE_MINI_MASK = False
    # Use smaller anchors because our image and objects are small
    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    #VALIDATION_STEPS = 5

    LEARNING_RATE = 0.01


def get_image_paths(image_path):
    images = []
    for foundImage in glob.glob(f'{image_path}/*.png'):
        images.append(os.path.abspath(foundImage))
    return images


class TrafficDataset(utils.Dataset):
    renders = []
    masks = []
    render_path = ""
    mask_path = ""

    def load_imageset(self, itemName, render_path):
        self.render_path = render_path
        self.mask_path = render_path+"/masks/"
        self.renders = get_image_paths(self.render_path)
        self.masks = get_image_paths(self.mask_path)
        self.add_class("stopsign", 1, "stopsign")
        for i in range(len(self.renders)):
            height, width, extra = cv2.imread(self.renders[i]).shape
            self.add_image(source=itemName, image_id=i, path=self.renders[i], width=width, height=height)


    def load_mask(self, image_id):
        mask = skimage.io.imread(self.masks[image_id])
        mask = skimage.color.rgb2gray(mask)
        globalthreshold = threshold_otsu(mask)
        mask = mask > globalthreshold

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]

def createModel(model_path, config, init_with, coco_path, train_path, test_path):
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=model_path)

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(coco_path, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    dataset_train = TrafficDataset(train_path)
    dataset_train.load_imageset("stopsign", train_path)
    dataset_train.prepare()


    dataset_val = TrafficDataset(test_path)
    dataset_val.load_imageset("stopsign", test_path)
    dataset_val.prepare()

    # train heads
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

    # train all
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=2,
                layers="all")


if __name__ == "__main__":
    coco_path = getWeights(r"C:\Users\anton\Documents\MaskRcnn\Mask_RCNN")
    config = TrafficConfig()
    createModel(r"C:\Users\anton\Documents\MaskRcnn\Mask_RCNN\model", config, "coco", coco_path,
                r"C:\Users\anton\Documents\MaskRcnn\Mask_RCNN\training",
                r"C:\Users\anton\Documents\MaskRcnn\Mask_RCNN\testing")
