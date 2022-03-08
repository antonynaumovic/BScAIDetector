from pycocotools.coco import COCO
import os
import json
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from imageGlobDetector import get_images_from_path

import Mask2Poly

def images_to_json(images_path):
    for image in get_images_from_path(images_path):
        print(image)
        image_path = images_path + "/" + image
        mask_path = images_path + r"/masks/" + image
        img, points = Mask2Poly.mask_to_poly(image_path,  mask_path, 0, 0)

if __name__ == "__main__":
    images_to_json(r"C:\Users\anton\Documents\MaskRcnn\Mask_RCNN\training")