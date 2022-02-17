import PySimpleGUI as Sg
import math
import random
from imageGlobDetector import *
from config import *
from imageTrainer import *


def main():
    layout_image_processor = [
        [Sg.Text("Please Select Images and Masks")],
        [Sg.Text("Image Folder"), Sg.In(get_config_image()[0], size=(50, 1), enable_events=True, key="-IMAGES-"),
         Sg.FolderBrowse()],
        [Sg.Text("Mask Folder"), Sg.In(get_config_image()[1], size=(50, 1), enable_events=True, key="-MASKS-"),
         Sg.FolderBrowse()],
        [Sg.Text("Training Folder"), Sg.In(get_config_image()[2], size=(50, 1), enable_events=True, key="-OUTPUT-"),
         Sg.FolderBrowse()],
        [Sg.Text("Testing Folder"), Sg.In(get_config_image()[3], size=(50, 1), enable_events=True, key="-TESTING-"),
         Sg.FolderBrowse()],
        [Sg.Text("Label"), Sg.In("stopsign", size=(50, 1), enable_events=True, key="-CURLABEL-")],
        [Sg.Text("% Test Split"), Sg.In("30", size=(50, 1), enable_events=True, key="-SPLIT-")],
        [Sg.Text("% Threshold"), Sg.In("1", size=(50, 1), enable_events=True, key="-THRESHOLD-")],
        [Sg.Button("Process")],
        [Sg.Button("Compress")],
        [Sg.ProgressBar(100, orientation='h', size=(50, 20), key='-PROGRESS-')]
    ]

    layout_model_trainer = [
        [Sg.Text("MODEL TRAINER")],
        [Sg.Text("Training Folder"), Sg.In(get_config_model()[0], size=(50, 1), enable_events=True, key="-TRAINING-"),
         Sg.FolderBrowse()],
        [Sg.Text("Labels Folder"), Sg.In(get_config_model()[1], size=(50, 1), enable_events=True, key="-LABELS-"),
         Sg.FolderBrowse()],
        [Sg.Text("Pre Trained Model"), Sg.In(get_config_model()[2], size=(50, 1), enable_events=True, key="-MODEL-"),
         Sg.FileBrowse(file_types=(("Text Files", "*.h5"),))],
        [Sg.Text("Batch Size"), Sg.In(get_config_model()[3], size=(50, 1), enable_events=True, key="-BATCH-")],
        [Sg.Text("Experiments Size"),
         Sg.In(get_config_model()[4], size=(50, 1), enable_events=True, key="-EXPERIMENTS-")],
        [Sg.Button("Process Model")],
    ]

    layout = [
        [Sg.Button('Image Processor', visible=False), Sg.Button('Model Trainer')],
        [Sg.Column(layout_image_processor, key='-VIEW1-'),
         Sg.Column(layout_model_trainer, visible=False, key='-VIEW2-')],
        [Sg.Button('Exit')]
    ]

    window = Sg.Window("AI Trainer", layout)
    while True:
        event, values = window.read()

        if event == "Image Processor":
            window['-VIEW2-'].update(visible=False)
            window['-VIEW1-'].update(visible=True)
            window['Image Processor'].update(visible=False)
            window['Model Trainer'].update(visible=True)

        if event == "Model Trainer":
            window['-VIEW1-'].update(visible=False)
            window['-VIEW2-'].update(visible=True)
            window['Model Trainer'].update(visible=False)
            window['Image Processor'].update(visible=True)

        if event == "Process Model":
            image_path = values["-TRAINING-"]
            labels_path = values["-LABELS-"]
            model_path = values["-MODEL-"]
            try:
                batch_size = int(values["-BATCH-"])
            except ValueError:
                batch_size = 5
            try:
                experiments = int(values["-EXPERIMENTS-"])
            except ValueError:
                experiments = 20

            print(image_path, labels_path, batch_size, experiments, model_path)
            set_config_model(image_path, labels_path, batch_size, experiments, model_path)
            train_model(image_path, labels_path, batch_size, experiments, model_path)

        if event == "Compress":
            window['-PROGRESS-'].UpdateBar(0)
            image_path = values["-IMAGES-"]

        if event == "Process":
            window['-PROGRESS-'].UpdateBar(0)
            image_path = values["-IMAGES-"]
            mask_path = values["-MASKS-"]
            output_path = values["-OUTPUT-"]
            testing_path = values["-TESTING-"]
            set_config_image(image_path, mask_path, output_path, testing_path)
            renders_array = get_images_from_path(image_path)
            masks_array = get_images_from_path(mask_path)
            matches = set(renders_array).intersection(masks_array)
            try:
                threshold = float(values["-THRESHOLD-"]) / 100
            except Exception as e:
                if type(e) == ValueError:
                    print("ERROR: Threshold not float")
                threshold = 0.05

            progress = 1
            verified_images = verify_images(matches, mask_path, threshold)
            train_percentage = math.floor(len(verified_images) / 100 * int(values["-SPLIT-"]))
            test_percentage = len(verified_images) - train_percentage
            test_train_split = ([0] * train_percentage) + ([1] * test_percentage)
            random.shuffle(test_train_split)
            imageIter = 0
            for image in verified_images:
                progress += 1 / len(verified_images) * 100
                if test_train_split[imageIter] == 0:
                    out_path = testing_path
                else:
                    out_path = output_path
                process_image(image_path + "/" + image, mask_path + "/" + image, out_path, values["-CURLABEL-"])
                window['-PROGRESS-'].UpdateBar(progress)
                imageIter += 1

        if event == "Exit" or event == Sg.WIN_CLOSED:
            break

    window.close()


if __name__ == "__main__":
    config_setup()
    main()
