import collections
from copy import deepcopy
import itertools
import math
from typing import List, Tuple, Union

import numpy as np
import torch
from ultralytics.yolo.utils.plotting import Annotator
from ultralytics.yolo.engine.results import Boxes


class ImageFragment:
    def __init__(self, img, center: Tuple[int, int]) -> None:
        self.im = img
        self.center = center
        self.shape = self.im.shape
        self.translate = torch.tensor(
            [self.center[0] - self.shape[0] / 2, self.center[1] - self.shape[1] / 2]
        ).cuda()

    def to_global_result(self, result):
        glob_res = deepcopy(result)
        for box in glob_res.boxes:
            box.xyxy[0, :2] += self.translate
            box.xyxy[0, 2:] += self.translate
            box.xywh[0, :2] += self.translate
        return glob_res


def get_intersection_area(box1, box2):
    """
    Get the intersection are of two boxes

    Args:
        box1 (torch.Tensor): first box
        box2 (torch.Tensor): second box

    Returns:
        float: area of the intersection
    """
    x_left = max(box1[0].item(), box2[0].item())
    y_top = max(box1[1].item(), box2[1].item())
    x_right = min(box1[2].item(), box2[2].item())
    y_bottom = min(box1[3].item(), box2[3].item())

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    return intersection_area


def get_iou(box1: torch.Tensor, box2: torch.Tensor):
    """
    Implement the intersection over union (IoU) between box1 and box2

    Arguments:
        box1 -- first box, numpy array with coordinates (ymin, xmin, ymax, xmax)
        box2 -- second box, numpy array with coordinates (ymin, xmin, ymax, xmax)
    """
    # ymin, xmin, ymax, xmax = box

    x_left = max(box1[0].item(), box2[0].item())
    y_top = max(box1[1].item(), box2[1].item())
    x_right = min(box1[2].item(), box2[2].item())
    y_bottom = min(box1[3].item(), box2[3].item())

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    box1_area = (box1[2].item() - box1[0].item()) * (box1[3].item() - box1[1].item())
    box2_area = (box2[2].item() - box2[0].item()) * (box2[3].item() - box2[1].item())

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_merged_box(boxes: Boxes) -> Boxes:
    """Merge all bboxes of a Boxes instance

    Args:
        boxes (Boxes): Bounding Boxes to merge

    Returns:
        Boxes: Merged bounding box
    """
    conf = max(b.conf for b in boxes)  # / len(boxes)
    cls_preds = {}
    for box in boxes:
        if box.cls.item() not in cls_preds:
            cls_preds[box.cls.item()] = box.conf.item()
        else:
            cls_preds[box.cls.item()] += box.conf.item()

    cls = max(cls_preds, key=cls_preds.get)

    return Boxes(
        torch.tensor(
            [
                min(box.xyxy[0][0] for box in boxes),
                min(box.xyxy[0][1] for box in boxes),
                max(box.xyxy[0][2] for box in boxes),
                max(box.xyxy[0][3] for box in boxes),
                conf,
                cls,
            ]
        ),
        orig_shape=boxes[0].orig_shape,
    )


def are_connected(box1: Boxes, box2: Boxes, intersect_th=0.3) -> bool:
    """
    Check whether two boxes are deemed to be 'connected'
    This is calculated by checking if the relative instersection area of one box is within a given threshold

    Args:
        box1 (Boxes): first box
        box2 (Boxes): second box
        intersect_th (float, optional): Relative intersection are to check against. Defaults to 0.3.

    Returns:
        bool: whether the boxes are connected
    """
    box1_data, box2_data = box1.xyxy[0], box2.xyxy[0]
    int_area = get_intersection_area(box1_data, box2_data)
    rel_b1 = int_area / (
        (box1_data[2].item() - box1_data[0].item())
        * (box1_data[3].item() - box1_data[1].item())
    )
    rel_b2 = int_area / (
        (box2_data[2].item() - box2_data[0].item())
        * (box2_data[3].item() - box2_data[1].item())
    )
    if max(rel_b1, rel_b2) > intersect_th and box1.cls == box2.cls:
        return True
    return False


def merge_boxes(boxes: List[Boxes], intersection_th=0.3) -> List[Boxes]:
    """Merge a set of boxes of different predictions to a single prediction

    Args:
        boxes (List[Boxes]): List of all prediction bboxes
        intersection_th (float, optional): Overlapping theshold to merge boxes. Defaults to 0.3.

    Returns:
        List[Boxes]: List of boxes without overlap / duplicates
    """
    result_set = []
    work_set = deepcopy(boxes)
    while work_set:
        connections = [
            [are_connected(b1, b2, intersect_th=intersection_th) for b2 in work_set]
            for b1 in work_set
        ]
        num_connections = [sum(conn) for conn in connections]

        if max(num_connections) <= 1:
            result_set.extend(work_set)
            break

        max_conn_idx = np.argmax(num_connections)
        merge_idx = [
            i for i, connected in enumerate(connections[max_conn_idx]) if connected
        ]
        boxes_to_merge = [work_set[i] for i in merge_idx]
        result_set.append(get_merged_box(boxes_to_merge))
        [work_set.remove(box) for box in boxes_to_merge]

    return result_set


def get_prediction_fragment(
    model, img, fragment_size: int = 1280, intersection_th=0.3, *args, **kwargs
) -> List[Boxes]:
    """Get the prediction using image fragmenting

    Args:
        model (_type_): model or list of models to predict on fragments
        img (_type_): whole image to predict on
        fragment_size (int, optional): Size of the individual image fragments. Defaults to 1280.
        intersection_th (float, optional): Threshold to use when merging prediction boxes. Defaults to 0.3.

    Returns:
        List[Boxes]: List of bounding box predictions on the whole image
    """
    imgs_x = math.ceil(img.shape[1] * 1.5 / fragment_size)
    imgs_y = math.ceil(img.shape[0] * 1.5 / fragment_size)
    h, w = img.shape[:2]

    dx = np.linspace(fragment_size / 2, w - fragment_size / 2, imgs_x)
    dy = np.linspace(fragment_size / 2, h - fragment_size / 2, imgs_y)

    fragments: List[ImageFragment] = []
    for x in dx:
        for y in dy:
            x_start = int(round(x - fragment_size / 2))
            x_end = int(round(x + fragment_size / 2))
            y_start = int(round(y - fragment_size / 2))
            y_end = int(round(y + fragment_size / 2))
            img_seg = img[y_start:y_end, x_start:x_end]
            fragments.append(
                ImageFragment(img=img_seg, center=(int(round(x)), int(round(y))))
            )
    if isinstance(model, collections.Iterable):
        results = itertools.chain(
            *(zip(fragments, m([frag.im for frag in fragments])) for m in model)
        )
    else:
        results = zip(fragments, model([frag.im for frag in fragments]))
    global_results = [frag.to_global_result(result) for frag, result in results]
    boxes = [box for res in global_results for box in res.boxes]
    merged_boxes = merge_boxes(boxes, intersection_th=intersection_th)
    return merged_boxes


def get_prediction_simple(model, img, *args, **kwargs) -> List[Boxes]:
    """
    Create a prediction on a single image - no image fragmentation

    Args:
        model (_type_): model or list of models
        img (_type_): image to predict on

    Returns:
        List[Boxes]: predictions on single image
    """
    if isinstance(model, collections.Iterable):
        results = [m(img) for m in model]
        boxes = [box for res in results for box in res.boxes]
        return merge_boxes(boxes)
    return model(img).boxes


def get_prediction(model, img, use_fragmentation: bool, **kwargs) -> List[Boxes]:
    """
    Create the predictions of a given model or models on a given image

    Args:
        model (_type_): model or list of models to use for prediction
        img (_type_): image to predict on
        use_fragmentation (bool): whether to use image fragmentation technique

    Returns:
        List[Boxes]: list of prediction boxes
    """
    if use_fragmentation:
        return get_prediction_fragment(model=model, img=img, **kwargs)
    return get_prediction_simple(model=model, img=img, **kwargs)
