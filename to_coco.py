import functools
import pathlib
import random
from typing import Union
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from pycocotools.coco import COCO
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import yaml
import shutil

DATA_ROOT = pathlib.Path(__file__).parent / "data" / "rdd2022" / "RDD2022"


def bbox_ltrb_to_ltwh(boxes_ltrb: Union[np.ndarray, torch.Tensor]):
    if not boxes_ltrb.size:
        return np.array([])
    cat = torch.cat if isinstance(boxes_ltrb, torch.Tensor) else np.concatenate
    assert boxes_ltrb.shape[-1] == 4
    return cat((boxes_ltrb[..., :2], boxes_ltrb[..., 2:] - boxes_ltrb[..., :2]), -1)


def bbox_ltrb_to_coco(boxes_ltrb: np.ndarray, width: int, height):
    if not boxes_ltrb.size:
        return np.array([])
    
    # move all points larger than the image to the image border
    boxes_ltrb[..., boxes_ltrb[...] < 0] = 0
    boxes_ltrb[..., 0] = np.clip(boxes_ltrb[..., 0], 0, width)
    boxes_ltrb[..., 1] = np.clip(boxes_ltrb[..., 1], 0, height)
    boxes_ltrb[..., 2] = np.clip(boxes_ltrb[..., 2], 0, width)
    boxes_ltrb[..., 3] = np.clip(boxes_ltrb[..., 3], 0, height)
    
    size_vec = np.array([width, height])
    coco_box = np.concatenate(
        (
            (boxes_ltrb[..., :2] + boxes_ltrb[..., 2:]) / (2 * size_vec),
            (boxes_ltrb[..., 2:] - boxes_ltrb[..., :2]) / size_vec,
        ),
        -1,
    )
    assert (coco_box[..., 0] + coco_box[..., 2] / 2 <= 1).all()
    assert (coco_box[..., 0] - coco_box[..., 2] / 2 >= 0).all()
    assert (coco_box[..., 1] + coco_box[..., 3] / 2 <= 1).all()
    assert (coco_box[..., 1] - coco_box[..., 3] / 2 >= 0).all()
    return coco_box


class COCO_Converter:
    class_names = ("__background__", "d00", "d10", "d20", "d40")

    def __init__(self, country, transform=None, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            data_dir: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.transform = transform
        self.keep_difficult = keep_difficult
        self.data_dir = (
            pathlib.Path(__file__).parent
            / "data"
            / "rdd2022"
            / "RDD2022"
            / country
            / "train"
        )
        self.image_folder = self.data_dir / "images"
        self.annotation_folder = self.data_dir / "annotations"
        self.image_ids = COCO_Converter._read_image_ids(self.image_folder)
        self.class_dict = {
            class_name: i for i, class_name in enumerate(self.class_names)
        }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        boxes, labels, is_difficult, im_info = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        boxes[:, [0, 2]] /= im_info["width"]
        boxes[:, [1, 3]] /= im_info["height"]
        image = self._read_image(image_id)
        sample = dict(
            image=image,
            boxes=boxes,
            labels=labels,
            width=im_info["width"],
            height=im_info["height"],
            image_id=image_id,
        )
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _read_image_ids(image_folder: pathlib.Path):
        return [int(f.stem[-6:]) for f in image_folder.glob("*.jpg")]

    # @staticmethod
    # def _get_train_test_split(imgage_dir: pathlib.Path, split_ratio: float, seed: int = 42) -> Tuple[List[str], List[str]]:
    #     """
    #     Get the Image IDs for the test and train sets

    #     Args:
    #         imgage_dir (pathlib.Path): directory containing the images
    #         split_ratio (float): percentage of images to be within the training set
    #         seed (int): random seed
    #     """
    #     train_split_file = imgage_dir / f'train_{str(split_ratio).replace(".", "")}_{seed}.txt'
    #     val_split_file = imgage_dir / f'val_{str(split_ratio).replace(".", "")}_{seed}.txt'
    #     if not (train_split_file.exists() and val_split_file.exists()):
    #         all_ids = COCO_Converter._read_image_ids(image_folder=imgage_dir)
    #         all_ids = [int(id_s[-6:]) for id_s in all_ids]
    #         split_idx = int(round(split_ratio*len(all_ids)))
    #         random.shuffle(all_ids)
    #         train, val = all_ids[:split_idx], all_ids[split_idx:]
    #         with train_split_file.open('w+') as f:
    #             f.writelines('\n'.join(train))
    #         with val_split_file.open('w+') as f:
    #             f.writelines('\n'.join(val))

    #     return train_split_file.read_text().split('\n'), val_split_file.read_text().split('\n')

    def _get_annotation(self, image_id):
        annotation_file = list(
            (self.annotation_folder / "xmls").glob(f"*{image_id:06d}.xml")
        )[0]
        ann_file = ET.parse(annotation_file)
        objects = ann_file.findall("object")

        size = ann_file.getroot().find("size")
        im_info = dict(
            height=int(size.find("height").text), width=int(size.find("width").text)
        )
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find("name").text.lower().strip()
            bbox = obj.find("bndbox")
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find("xmin").text) - 1
            y1 = float(bbox.find("ymin").text) - 1
            x2 = float(bbox.find("xmax").text) - 1
            y2 = float(bbox.find("ymax").text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = obj.find("difficult").text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(is_difficult, dtype=np.uint8),
            im_info,
        )

    def _read_image(self, image_id):
        image_file = self.image_folder / f"{image_id}.jpg"
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def get_annotations_as_coco(self) -> COCO:
        """
        Returns bounding box annotations in COCO dataset format
        """
        coco_anns = {
            "annotations": [],
            "images": [],
            "licences": [{"name": "", "id": 0, "url": ""}],
            "categories": [
                {"name": cat, "id": i + 1, "supercategory": ""}
                for i, cat in enumerate(self.class_names)
            ],
        }
        ann_id = 1
        for idx in tqdm(range(len(self))):
            image_id = self.image_ids[idx]
            boxes_ltrb, labels, _, im_info = self._get_annotation(image_id)
            boxes_ltwh = bbox_ltrb_to_coco(
                boxes_ltrb, width=im_info["width"], height=im_info["height"]
            )
            coco_anns["images"].append({"id": image_id, **im_info})
            for box, label in zip(boxes_ltwh, labels):
                box = box.tolist()
                area = box[-1] * box[-2]

                coco_anns["annotations"].append(
                    {
                        "bbox": box,
                        "area": area,
                        "category_id": int(label),
                        "image_id": image_id,
                        "id": ann_id,
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1
        coco_anns["annotations"].sort(key=lambda x: x["image_id"])
        coco_anns["images"].sort(key=lambda x: x["id"])
        coco = COCO()
        coco.dataset = coco_anns
        coco.createIndex()
        yaml_file = self.data_dir.parent / "coco.yaml"
        yaml.dump(coco_anns, yaml_file.open("w"))
        return coco

    def export_img(self, img_id, export_img_dir, export_ann_dir):
        img_file = list(self.image_folder.glob(f"*_{img_id:06d}.jpg"))[0]
        shutil.copyfile(img_file, export_img_dir / img_file.name)

        boxes_ltrb = self._get_annotation(img_id)
        boxes_ltrb, labels, _, im_info = self._get_annotation(img_id)
        boxes_ltwh = bbox_ltrb_to_coco(
            boxes_ltrb, width=im_info["width"], height=im_info["height"]
        )

        annotation_file = export_ann_dir / f"{img_file.stem}.txt"
        annotation_file.write_text(
            "\n".join(
                f"{label - 1} {shape[0]} {shape[1]} {shape[2]} {shape[3]}"
                for label, shape in zip(labels, boxes_ltwh)
            ) + "\n"
        )

    def export_yolov8(
        self,
        train_split: float = 0.9,
        dst: pathlib.Path = pathlib.Path(__file__).parent / 'yolo' / "datasets" / "rdd2022",
    ):
        all_ids = self.image_ids
        random.shuffle(self.image_ids)
        split_idx = int(round(len(self.image_ids) * train_split))
        datasets = {"train": all_ids[:split_idx], "val": all_ids[split_idx:]}
        export_dir = dst

        for key, ids in datasets.items():
            export_img_dir = export_dir / key / "images"
            export_img_dir.mkdir(exist_ok=True, parents=True)
            export_ann_dir = export_dir / key / "labels"
            export_ann_dir.mkdir(exist_ok=True, parents=True)

            process_map(
                functools.partial(
                    self.export_img,
                    export_img_dir=export_img_dir,
                    export_ann_dir=export_ann_dir,
                ),
                ids,
                max_workers=10,
                chunksize=10,
            )
            # [self.export_img(i, export_img_dir=export_img_dir, export_ann_dir=export_ann_dir) for i in ids]


if __name__ == "__main__":
    conv = COCO_Converter("Norway")
    conv.export_yolov8()
