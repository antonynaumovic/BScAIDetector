import numpy as np
import time
import os
import cv2
import argparse
import glob
import PySimpleGUI as sg
import configparser

from imageTrainer import train_model

def show_image(image):
    cv2.imshow('image',image)
    c = cv2.waitKey()
    if c >= 0 : return -1
    return 0

def main_gui():
    logsMessage = ""
    layout = [
        [sg.Text("Please Select Images and Masks")],
        [sg.Text("Image Folder"), sg.In(get_config()[0], size=(50, 1), enable_events=True, key="-IMAGES-"), sg.FolderBrowse()],
        [sg.Text("Mask Folder"), sg.In(get_config()[1], size=(50, 1), enable_events=True, key="-MASKS-"), sg.FolderBrowse()],
        [sg.Text("Output Folder"), sg.In(get_config()[2], size=(50, 1), enable_events=True, key="-OUTPUT-"), sg.FolderBrowse()],
        [sg.Text("% Threshold"), sg.In("5", size=(50, 1), enable_events=True, key="-THRESHOLD-")],
        [sg.Button("Process")],
        [sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS-')]
    ]

    window = sg.Window("Images", layout)
    while True:
        event, values = window.read()
        # End program if user closes window or
        # presses the OK button

        if event == "Process":
            imagePath = values["-IMAGES-"]
            maskPath = values["-MASKS-"]
            outputPath = values["-OUTPUT-"]
            set_config(imagePath, maskPath, outputPath)
            rendersArray = get_images_from_path(imagePath)
            masksArray = get_images_from_path(maskPath)
            matches = set(rendersArray).intersection(masksArray)
            try:
                threshold = float(values["-THRESHOLD-"]) / 100
            except Exception as e:
                if type(e) == ValueError:
                    print("ERROR: Threshold not float")
                threshold = 0.05

            progress = 1
            verifiedImages = verify_images(matches, maskPath, threshold)
            for image in verifiedImages:
                progress += 1/len(verifiedImages) * 100
                process_image(imagePath+"/"+image, maskPath+"/"+image, outputPath)
                window['-PROGRESS-'].UpdateBar(progress)

        if event == "OK" or event == sg.WIN_CLOSED:
            break

    window.close()



def verify_images(imageArray, maskPath, threshold):
    matchesIter = imageArray.copy()
    for image in matchesIter:
        if not verify_image(maskPath + "/" + image, threshold):
            imageArray.remove(image)
            print(f"Removing: {image} for not enough of object")
    return imageArray

def main():
    main_gui()

def set_config(renderPath, maskPath, outputPath):
    config = configparser.ConfigParser()
    with open("cookies.ini", "w+") as ini:
        config.add_section('paths')
        config.set('paths', 'image', renderPath)
        config.set('paths', 'mask', maskPath)
        config.set('paths', 'output', outputPath)
        config.write(ini)

def get_config():
    config = configparser.ConfigParser()
    config.read('cookies.ini')
    imagesPath = config['paths']['image']
    maskPath = config['paths']['mask']
    outputPath = config['paths']['output']
    return imagesPath, maskPath, outputPath

def config_setup():
    config = configparser.ConfigParser()
    with open("cookies.ini", "r+") as ini:
        if len(ini.readlines()) == 0:
            config.add_section('paths')
            config.set('paths', 'image', '')
            config.set('paths', 'mask', '')
            config.set('paths', 'output', '')
            config.write(ini)


def get_images_from_path(imagePath):
    renders = []
    masks = []
    for foundImage in glob.glob(f'{imagePath}/*.png'):
        renders.append(os.path.basename(foundImage))
    return renders


def process_image(renderPath, maskPath, outputPath):
    im = cv2.imread(maskPath)
    #cv2.imshow("Mask", im)
    im_render = cv2.imread(renderPath)
    #cv2.imshow("Colour", im_render)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    retries, threshold = cv2.threshold(imgray, 120, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #img = cv2.drawContours(im_render, contours, -1, (255,255,0), 2)

    areas = [cv2.contourArea(cont) for cont in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x,y,w,h = cv2.boundingRect(cnt)
    #cv2.rectangle(im_render,(x,y),(x+w,y+h),(0,255,0),2)
    crop_img = im_render[y:y + h, x:x + w]
    #show_image(crop_img)
    cv2.imwrite(outputPath+"/"+os.path.basename(renderPath),crop_img)


def verify_image(maskPath, threshold):
    img = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
    whitePixels = np.sum(img == 255)
    totalPixels = img.shape[0]*img.shape[1]
    whitePercent = (whitePixels/totalPixels) * 100
    if whitePercent < threshold:
        return False
    else:
        return True



if __name__ == "__main__":
    config_setup()
    main()