import albumentations as A
import random
import numpy as np


class CustomAlbumentations:
    def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None

        T = [
            A.Blur(p=0.25),
            A.MedianBlur(p=0.25),
            A.ToGray(p=0.25),
            A.CLAHE(p=0.25),
            A.RandomBrightnessContrast(p=0.25),
            A.RandomGamma(p=0.25),
            A.ImageCompression(quality_lower=75, p=0.1)
            ]  # transforms
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels['img']
        cls = labels['cls']
        if len(cls):
            labels['instances'].convert_bbox('xywh')
            labels['instances'].normalize(*im.shape[:2][::-1])
            bboxes = labels['instances'].bboxes
            # TODO: add supports of segments and keypoints
            if self.transform and random.random() < self.p:
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new['class_labels']) > 0:  # skip update if no bbox in new im
                    labels['img'] = new['image']
                    labels['cls'] = np.array(new['class_labels'])
                    bboxes = np.array(new['bboxes'])
            labels['instances'].update(bboxes=bboxes)
        return labels