import json
from imageGlobDetector import get_images_from_path
import Mask2Poly
import datetime
from PIL import Image
import os
import numpy as np
from pycococreatortools import pycococreatortools

def images_to_json(images_path, json_name, output_path):
    info = {
        "year": 2022,
        "version": "1.0",
        "description": "3D Rendered Dataset",
        "contributor": "Antony Naumovic",
        "url": "",
        "date_created": datetime.datetime(2022, 3, 8).strftime("%Y/%m/%d"),
    }
    licenses = [
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License",
        }
    ]

    categories = [
        {
            "id": 1,
            "name": "stopsign",
            "supercategory": "traffic",
        }
    ]
    images = []
    annotations = []
    image_id = 1
    segmentation_id = 1

    coco_output = {
        "info": info,
        "licenses": licenses,
        "categories": categories,
        "images": [],
        "annotations": []
    }

    for image in get_images_from_path(images_path):
        # image_json = {
        #     "id": image_id,
        #     "license": 0,
        #     "file_name": image,
        #     "height": 1024,
        #     "width": 1024,
        #     "date_captured": datetime.datetime(2022, 3, 8).strftime("%Y/%m/%d"),
        #     "flickr_url": "",
        #     "coco_url":"",
        # }
        # mask_path = images_path + r"/masks/" + image
        # annotation, points = Mask2Poly.mask_to_poly(mask_path, 0, image_id)
        # images.append(image_json)
        # annotations.append(annotation)
        # image_id += 1

        image_path = images_path + "/" + image

        image_open = Image.open(image_path)
        image_info = pycococreatortools.create_image_info(
            image_id, os.path.basename(image_path), image_open.size)
        coco_output["images"].append(image_info)

        mask_path = images_path + r"/masks/" + image
        category_info = {'id': 1, 'is_crowd': 0}
        binary_mask = np.asarray(Image.open(mask_path).convert('1')).astype(np.uint8)
        annotation_info = pycococreatortools.create_annotation_info(
            segmentation_id, image_id, category_info, binary_mask,
            image_open.size, tolerance=2)

        coco_output["annotations"].append(annotation_info)
        image_id += 1
        segmentation_id += 1

    # full_json = {
    #     "info": info,
    #     "images": images,
    #     "annotations": annotations,
    #     "licenses": licenses,
    #     "categories": categories,
    #     }
    # json_data = json.dumps(full_json)
    # print(json_data)
    # with open(output_path + "/" + json_name + ".json", "w+") as json_file:
    #     json.dump(json_data, json_file)

    with open(output_path + "/" + json_name + ".json", 'w') as output_json_file:
        json.dump(coco_output, output_json_file)


if __name__ == "__main__":
    images_to_json(r"C:\Users\anton\Documents\DetectronDataset\training", "training",
                   r"C:\Users\anton\Documents\DetectronDataset")
    images_to_json(r"C:\Users\anton\Documents\DetectronDataset\testing", "testing",
                   r"C:\Users\anton\Documents\DetectronDataset")
