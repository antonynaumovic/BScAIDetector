import numpy as np
import os
import cv2
import glob
import pascal_voc_writer
from pathlib import Path


def show_image(image):
    cv2.imshow('image', image)
    c = cv2.waitKey()
    if c >= 0:
        return -1
    return 0


def verify_images(image_array, mask_path, threshold):
    matches_iter = image_array.copy()
    for image in matches_iter:
        if not verify_image(mask_path + "/" + image, threshold):
            image_array.remove(image)
            print(f"Removing: {image} for not enough of object")
    return image_array


def get_images_from_path(image_path):
    renders = []
    for foundImage in glob.glob(f'{image_path}/*.png'):
        renders.append(os.path.basename(foundImage))
    return renders


def compress_image(image_path):
    img = cv2.imread(image_path)
    cv2.imwrite(image_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def process_image(render_path, mask_path, output_path, image_label):
    im = cv2.imread(mask_path)
    image_render = cv2.imread(render_path)
    image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    retries, threshold = cv2.threshold(image_gray, 120, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cont) for cont in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnt)
    # crop_img = image_render[y:y + h, x:x + w]
    height, width, extra = im.shape
    final_img_output = output_path + "/" + os.path.basename(render_path)
    cv2.imwrite(output_path + "/" + os.path.basename(render_path), image_render)
    writer = pascal_voc_writer.Writer(final_img_output, height, width)
    if w > x:
        x, w = w, x
    if h > y:
        y, h = h, y
    writer.addObject(image_label, w, h, x, y)
    xml_output_path = str(Path(output_path).parent.absolute().resolve())
    writer.save(xml_output_path + "/annotations/" + os.path.splitext(os.path.basename(render_path))[0] + ".xml")


def verify_image(mask_path, threshold):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    white_pixels = np.sum(img == 255)
    total_pixels = img.shape[0] * img.shape[1]
    white_percent = (white_pixels / total_pixels) * 100
    if white_percent < threshold:
        return False
    return True
