import functools
import json
import pathlib

import yaml
from trainer import CustomTrainer
from ultralytics import YOLO
from ultralytics.yolo.engine.model import TASK_MAP
from val import CustomValidator
from utils import CLASS_DICT

def freeze_layers(trainer, num_layers):
    freeze = [f'model.{x}.' for x in range(num_layers)]  # layers to freeze 
    for k, v in trainer.model.named_parameters(): 
        v.requires_grad = True  # train all layers 
        if any(x in k for x in freeze): 
            # print(f'freezing {k}') 
            v.requires_grad = False 

def train():
    hyperparams = yaml.load(
        (pathlib.Path(__file__).parent / "hyperparams.yaml").open("r"),
        Loader=yaml.Loader,
    )
    print(json.dumps(hyperparams, indent=4))
    coco_file = pathlib.Path(__file__).parent / "config.yaml"

    
    TASK_MAP["detect"][1] = CustomTrainer
    TASK_MAP["detect"][2] = CustomValidator
    model = YOLO(hyperparams.pop('model'), task="detect")
    # model.val(data=str(coco_file))
    model.add_callback('on_train_start', func=functools.partial(freeze_layers, num_layers=hyperparams.pop("freeze", 0)))
    model.train(data=str(coco_file), **hyperparams)


if __name__ == "__main__":
    train()
