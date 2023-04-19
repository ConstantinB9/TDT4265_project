import json
import pathlib

import yaml
from trainer import CustomTrainer
from ultralytics import YOLO
from ultralytics.yolo.engine.model import TASK_MAP


def train():
    hyperparams = yaml.load(
        (pathlib.Path(__file__).parent / "hyperparams.yaml").open("r"),
        Loader=yaml.Loader,
    )
    print(json.dumps(hyperparams, indent=4))
    coco_file = pathlib.Path(__file__).parent / "config.yaml"

    TASK_MAP["detect"][1] = CustomTrainer
    model = YOLO(hyperparams.pop("model"), task="detect")
    model.tune(data=str(coco_file), gpu_per_trial=1, train_args=dict(cache=True))


if __name__ == "__main__":
    train()
