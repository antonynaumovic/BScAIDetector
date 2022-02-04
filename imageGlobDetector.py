import numpy as np
import os
import cv2
import glob
import configparser


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


def set_config(render_path, mask_path, output_path):
    config = configparser.ConfigParser()
    with open("cookies.ini", "w+") as ini:
        config.add_section('paths')
        config.set('paths', 'image', render_path)
        config.set('paths', 'mask', mask_path)
        config.set('paths', 'output', output_path)
        config.write(ini)


def get_config():
    config = configparser.ConfigParser()
    config.read('cookies.ini')
    images_path = config['paths']['image']
    mask_path = config['paths']['mask']
    output_path = config['paths']['output']
    return images_path, mask_path, output_path


def config_setup():
    config = configparser.ConfigParser()
    with open("cookies.ini", "r+") as ini:
        if len(ini.readlines()) == 0:
            config.add_section('paths')
            config.set('paths', 'image', '')
            config.set('paths', 'mask', '')
            config.set('paths', 'output', '')
            config.write(ini)


def get_images_from_path(image_path):
    renders = []
    for foundImage in glob.glob(f'{image_path}/*.png'):
        renders.append(os.path.basename(foundImage))
    return renders


def process_image(render_path, mask_path, output_path):
    im = cv2.imread(mask_path)
    image_render = cv2.imread(render_path)
    image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    retries, threshold = cv2.threshold(image_gray, 120, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cont) for cont in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    crop_img = image_render[y:y + h, x:x + w]
    cv2.imwrite(output_path + "/" + os.path.basename(render_path), crop_img)


def verify_image(mask_path, threshold):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    white_pixels = np.sum(img == 255)
    total_pixels = img.shape[0]*img.shape[1]
    white_percent = (white_pixels/total_pixels) * 100
    if white_percent < threshold:
        return False
    return True

