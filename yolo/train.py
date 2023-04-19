import functools
import json
import pathlib
import shutil
from typing import Tuple

import cv2
import yaml
from tqdm.contrib.concurrent import process_map
from trainer import CustomTrainer
from ultralytics import YOLO
from ultralytics.yolo.engine.model import TASK_MAP
from val import CustomValidator
import torch


def resize_move(
    src: pathlib.Path, dst_root: pathlib.Path, size: Tuple[int, int]
) -> None:
    assert src.exists()
    dst = dst_root / src.name
    img = cv2.imread(str(src))
    resize = cv2.resize(img, size)
    cv2.imwrite(str(dst), resize)


def resize_images(
    original_dataset_root: pathlib.Path,
    work_dataset_root: pathlib.Path,
    image_size: Tuple[int, int],
):
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
