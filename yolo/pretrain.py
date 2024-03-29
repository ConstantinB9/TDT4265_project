import functools
import json
import pathlib
import shutil
from typing import Tuple

import cv2
import yaml
from tqdm.contrib.concurrent import process_map
from ultralytics import YOLO
from ultralytics.yolo.engine.model import TASK_MAP

from pretrainer import CustomPreTrainer
from val import CustomValidator


def pretrain():
    """
    Script to run pretraining on the whole RDD2022 dataset excluding Norway
    """
    hyperparams = yaml.load(
        (pathlib.Path(__file__).parent / "prehyperparams.yaml").open("r"),
        Loader=yaml.Loader,
    )
    print(json.dumps(hyperparams, indent=4))
    coco_file = pathlib.Path(__file__).parent / "preconfig.yaml"

    TASK_MAP["detect"][1] = CustomPreTrainer
    TASK_MAP["detect"][2] = CustomValidator
    model = YOLO(hyperparams.pop("model"), task="detect")
    # model.val(data=str(coco_file))
    model.train(data=str(coco_file), **hyperparams)


if __name__ == "__main__":
    pretrain()
