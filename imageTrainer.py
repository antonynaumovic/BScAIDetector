from imageai.Detection.Custom import DetectionModelTrainer


def train_model(imagePath, imageLabels, batchSize, numExperiments):

    imageTrainer = DetectionModelTrainer()
    imageTrainer.setModelTypeAsYOLOv3()
    imageTrainer.setDataDirectory(data_directory="")

    imageTrainer.setTrainConfig(object_names_array=["person hardhat"], batch_size=4, num_experiments=20,
                                train_from_pretrained_model="yolo.h5")

    imageTrainer.trainModel()
