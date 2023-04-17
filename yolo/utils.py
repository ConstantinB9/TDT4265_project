import os
import pathlib
import xml.etree.ElementTree as ET
from typing import Union

import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageOps
from ultralytics.yolo.data.utils import IMG_FORMATS, exif_size

CLASS_DICT = {"d00": 0, "d10": 1, "d20": 2, "d40": 3}


def load_annotations(lb_file):
    ann_file = ET.parse(pathlib.Path(lb_file))
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
        if class_name in CLASS_DICT:
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_DICT[class_name])
            is_difficult_str = obj.find("difficult").text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
    boxes = bbox_ltrb_to_coco(np.array(boxes), **im_info) if boxes else np.zeros((0, 4))
    boxes = boxes.reshape(-1, 4)
    return (
        boxes,
        np.array(labels, dtype=np.int64),
        np.array(is_difficult, dtype=np.uint8),
        im_info,
    )


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix, keypoint, num_cls, nkpt, ndim = args
    # number (missing, found, empty, corrupt), message, segments, keypoints
    nm, nf, ne, nc, msg, segments, keypoints = 0, 0, 0, 0, "", [], None
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        shape = (shape[1], shape[0])  # hw
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in ("jpg", "jpeg"):
            with open(im_file, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(
                        im_file, "JPEG", subsampling=0, quality=100
                    )
                    msg = (
                        f"{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved"
                    )

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found

            boxes, classes, diff, info = load_annotations(lb_file)
            lb = np.concatenate((classes.reshape(-1, 1), boxes), 1)

            nl = len(lb)
            if nl:
                if keypoint:
                    assert lb.shape[1] == (
                        5 + nkpt * ndim
                    ), f"labels require {(5 + nkpt * ndim)} columns each"
                    assert (
                        lb[:, 5::ndim] <= 1
                    ).all(), "non-normalized or out of bounds coordinate labels"
                    assert (
                        lb[:, 6::ndim] <= 1
                    ).all(), "non-normalized or out of bounds coordinate labels"
                else:
                    assert (
                        lb.shape[1] == 5
                    ), f"labels require 5 columns, {lb.shape[1]} columns detected"
                    assert (
                        lb[:, 1:] <= 1
                    ).all(), f"non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}"
                    assert (lb >= 0).all(), f"negative label values {lb[lb < 0]}"
                # All labels
                max_cls = int(lb[:, 0].max())  # max label count
                assert max_cls <= num_cls, (
                    f"Label class {max_cls} exceeds dataset class count {num_cls}. "
                    f"Possible class labels are 0-{num_cls - 1}"
                )
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f"{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed"
            else:
                ne = 1  # label empty
                lb = (
                    np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32)
                    if keypoint
                    else np.zeros((0, 5), dtype=np.float32)
                )
        else:
            nm = 1  # label missing
            lb = (
                np.zeros((0, (5 + nkpt * ndim)), dtype=np.float32)
                if keypoint
                else np.zeros((0, 5), dtype=np.float32)
            )
        if keypoint:
            keypoints = lb[:, 5:].reshape(-1, nkpt, ndim)
            if ndim == 2:
                kpt_mask = np.ones(keypoints.shape[:2], dtype=np.float32)
                kpt_mask = np.where(keypoints[..., 0] < 0, 0.0, kpt_mask)
                kpt_mask = np.where(keypoints[..., 1] < 0, 0.0, kpt_mask)
                keypoints = np.concatenate(
                    [keypoints, kpt_mask[..., None]], axis=-1
                )  # (nl, nkpt, 3)
        lb = lb[:, :5]
        return im_file, lb, shape, segments, keypoints, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f"{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}"
        return [None, None, None, None, None, nm, nf, ne, nc, msg]


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
