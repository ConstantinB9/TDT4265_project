import functools
import io
import json
import pathlib
from typing import Tuple
from ultralytics import YOLO
from ultralytics.yolo.engine.model import TASK_MAP
import yaml
import cv2
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import shutil
from skimage import io
import torch
import torchvision.transforms as transforms
from PIL import Image

from trainer import CustomTrainer


def resize_move(
    src: pathlib.Path, dst_root: pathlib.Path, size: Tuple[int, int]
) -> None:
    assert src.exists()
    dst = dst_root / src.name
    img = cv2.imread(str(src))
    resize = cv2.resize(img, size)
    cv2.imwrite(str(dst), resize)

def resize_images(original_dataset_root: pathlib.Path, work_dataset_root: pathlib.Path, image_size: Tuple[int, int]):
        print("Resizing Images")
        work_dataset_root.mkdir(exist_ok=True, parents=True)
        for set in ("train", "val", "test"):
            set_root = original_dataset_root / set
            if not set_root.exists():
                continue
            img_folder = work_dataset_root / set / "images"
            img_folder.mkdir(exist_ok=True, parents=True)
            process_map(
                functools.partial(resize_move, size=image_size, dst_root=img_folder),
                list((set_root / "images").iterdir()),
                chunksize=10,
                max_workers=5,
                desc=f"Risizing images of {set}",
            )
            print("Copying labels")
            shutil.copytree(
                str(set_root / "labels"),
                str(work_dataset_root / set / "labels"),
                dirs_exist_ok=True,
            )


def train(hyperparams):
    print(json.dumps(hyperparams, indent=4))
    image_size = (hyperparams["image_size"], hyperparams["image_size"])

    coco_file = pathlib.Path(__file__).parent / "config.yaml"
    config = yaml.load(coco_file.open("r"), Loader=yaml.Loader)
    datasets_root = pathlib.Path(__file__).parent / "datasets"
    original_dataset = datasets_root / config["path"]

    work_dataset_root = datasets_root / (
        config["path"] + f"_{image_size[0]}x{image_size[1]}"
    )
    if not work_dataset_root.exists():
        resize_images(original_dataset_root=original_dataset, work_dataset_root=work_dataset_root, image_size=image_size)

    config["path"] = work_dataset_root.name
    new_config = coco_file.parent / f"config_{image_size[0]}x{image_size[1]}.yaml"
    yaml.dump(config, new_config.open("w"))

    runs_root = pathlib.Path(__file__).parent.parent / "runs"
    # TASK_MAP["detect"][1] = CustomTrainer
    model = YOLO(hyperparams["model"])
    model.train(data=str(new_config),
                epochs=hyperparams["epochs"],
                cache = False,
                resume=False,
                batch = 32,
                imgsz = hyperparams["image_size"],
                optimizer = hyperparams["optimizer"],
                cos_lr = hyperparams["cos_lr"],
                lr0 = hyperparams["lr0"],
                lrf = hyperparams["lrf"],
                momentum = hyperparams["momentum"],
                weight_decay = hyperparams["weight_decay"],
                box = hyperparams["box"],
                cls = hyperparams["cls"],
                dfl = hyperparams["dfl"]
                )
    metrics = model.val()
    print(metrics)
    # success = model.export(format="onnx")
    return metrics.results_dict


if __name__ == "__main__":
    default_params = {
        "model": "yolov8n.pt",
        "epochs": 100,
        "image_size": 640,
        "optimizer": "SGD",
        "cos_lr": False,
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
    }
    train(default_params)
