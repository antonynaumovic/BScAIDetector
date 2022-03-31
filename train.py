from utils import *
from detectron2.utils.logger import setup_logger
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

setup_logger()

config_save_path = r"/home/ant/Documents/Detectron2/Dataset/cfg.pickle"
config_path = r"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
checkpoint = r"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
output_path = r"/home/ant/Documents/Detectron2/Dataset/output/"
num_classes = 1

device = "cuda"

train_dataset = r"Traffic_train"
train_path = r"/home/ant/Documents/Detectron2/Dataset/training/"
train_json_path = r"/home/ant/Documents/Detectron2/Dataset/training.json"

test_dataset = r"Traffic_test"
test_path = r"/home/ant/Documents/Detectron2/Dataset/testing/"
test_json_path = r"/home/ant/Documents/Detectron2/Dataset/testing.json"


register_coco_instances(name=train_dataset, metadata={},
                        json_file=train_json_path, image_root=train_path)
register_coco_instances(name=test_dataset, metadata={},
                        json_file=test_json_path, image_root=test_path)


def main():
    config = get_train_config(config_path, checkpoint, train_dataset, test_dataset, num_classes, device, output_path)
    with open(config_save_path, 'wb+') as config_file:
        pickle.dump(config, config_file, protocol=pickle.HIGHEST_PROTOCOL)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(config)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()
