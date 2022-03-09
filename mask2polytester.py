from pycocotools.coco import COCO
import os
import json
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from imageGlobDetector import get_images_from_path

import Mask2Poly

def images_to_json(images_path):
    info = {
        "year": "2022",
        "version": "1.0",
        "description": "3D Rendered Dataset",
        "contributor": "Antony Naumovic"
    }
    licenses = {
          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
          "id": 0,
          "name": "Attribution-NonCommercial-ShareAlike License"
        }

    categories = [
        {
            "id": 0,
            "name":"stopsign",
            "supercategory":"traffic"
        }
    ]
    images = []
    annotations = []
    cur_image_id = 0
    for image in get_images_from_path(images_path):
        print(image)
        image_json = {
            "id":cur_image_id,
            "license":0,
            "file_name":image,
            "height":1024,
            "width":1024,
            "date_captured": None
        }
        image_path = images_path + "/" + image
        mask_path = images_path + r"/masks/" + image
        annotation, points = Mask2Poly.mask_to_poly(image_path,  mask_path, 0, cur_image_id)
        images.append(image_json)
        annotations.append(annotation)
        cur_image_id += 1

    full_json = {}
    full_json["info"] = info
    full_json["licenses"] = licenses
    full_json["categories"] = categories
    full_json["images"] = images
    full_json["annotations"] = annotations
    json_data = json.dumps(full_json)
    print(json_data)

if __name__ == "__main__":
    images_to_json(r"C:\Users\anton\Documents\MaskRcnn\Mask_RCNN\training")