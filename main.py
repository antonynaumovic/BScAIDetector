import PySimpleGUI as sg
from imageGlobDetector import *

def main():
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

if __name__ == "__main__":
    config_setup()
    main()