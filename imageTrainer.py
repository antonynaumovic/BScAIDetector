from imageai.Detection.Custom import DetectionModelTrainer


def train_model(image_path, image_labels, batch_size, num_experiments, pre_train_model):
    if "/" not in image_labels and "\\" not in image_labels:
        image_labels = image_labels.split()

    print(type(image_labels))
    image_trainer = DetectionModelTrainer()
    image_trainer.setModelTypeAsYOLOv3()
    image_trainer.setDataDirectory(data_directory=image_path)
    image_trainer.setTrainConfig(object_names_array=image_labels, batch_size=batch_size,
                                 num_experiments=num_experiments,
                                 train_from_pretrained_model=pre_train_model)
    image_trainer.trainModel()
