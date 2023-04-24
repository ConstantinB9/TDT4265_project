import json
import pathlib

import yaml
from trainer import CustomTrainer
from ultralytics import YOLO
from ultralytics.yolo.engine.model import TASK_MAP
from val import CustomValidator
from utils import CLASS_DICT

def train():
    hyperparams = yaml.load(
        (pathlib.Path(__file__).parent / "hyperparams.yaml").open("r"),
        Loader=yaml.Loader,
    )
    print(json.dumps(hyperparams, indent=4))
    coco_file = pathlib.Path(__file__).parent / "config.yaml"

    
    TASK_MAP["detect"][1] = CustomTrainer
    TASK_MAP["detect"][2] = CustomValidator
    
    model = YOLO(hyperparams.pop("model"), task="detect")
    # model.val(data=str(coco_file))
    model.train(data=str(coco_file), **hyperparams)


if __name__ == "__main__":
    train()
