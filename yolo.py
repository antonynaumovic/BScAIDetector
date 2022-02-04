import numpy as np
import time
import os
import cv2
import argparse
from tkinter import filedialog


argparser = argparse.ArgumentParser()
argparser.add_argument("-i", "--image", required=True, help="Image Input Path")

argparser.add_argument("-y", "--yolo", required=True, help="Yolo Directory")

argparser.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum Confidence")

argparser.add_argument("-t", "--threshold", type=float, default=0.3, help="Non-Maxima Threshold")

args = vars(argparser.parse_args())

usePreviousPaths = False

if usePreviousPaths == False:
    labelsPath = filedialog.askdirectory()
    weightsPath = filedialog.askdirectory()
    configPath = filedialog.askdirectory()
    with open("paths.txt", "w+") as pathFile:
        pathFile.write(labelsPath + "\n")
        pathFile.write(weightsPath + "\n")
        pathFile.write(configPath + "\n")

else:
    with open("paths.txt", "r+") as pathFile:
        labelsPath = pathFile.read(0)
        weightsPath = pathFile.read(1)
        configPath = pathFile.read(2)

labels = open(labelsPath).read().strip().split("\n")

np.random.seed(0)
colors = np.random.randint(0,255,size=(len(labels),3), dtype="uint8")
nn = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

