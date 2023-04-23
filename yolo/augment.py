import cv2
import numpy as np
from ultralytics.yolo.data.augment import BaseTransform


class CropFragment(BaseTransform):
    def __init__(self, fragment_size:int = 1280, gravitate_to_labels: bool = False, resize_shape: int = 640) -> None:
        super().__init__()
        self.fragment_size = fragment_size
        self.gravitate_to_labels = gravitate_to_labels
        self.resize_shape = resize_shape
        
    def __call__(self, labels):
        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format='xyxy')
        instances.denormalize(*img.shape[:2][::-1])

        
        xmin, xmax = self.fragment_size / 2, img.shape[0] - self.fragment_size / 2
        ymin, ymax = self.fragment_size / 2, img.shape[1] - self.fragment_size / 2
        
        cx = int(round(np.random.uniform(xmin, xmax, 1)[0]))
        cy = int(round(np.random.uniform(ymin, ymax, 1)[0]))
        
        img = img[
            int(cx-self.fragment_size/2):int(cx+self.fragment_size/2),
            int(cy-self.fragment_size/2):int(cy+self.fragment_size/2)
                  ]
        img = cv2.resize(img, (self.resize_shape, self.resize_shape))
        
        dx = cx - self.fragment_size / 2
        dy = cy - self.fragment_size / 2
        
        boxes = instances.bboxes
        
        boxes -= np.array([dy, dx]*2)
        instances.update(boxes)
        instances.clip(self.fragment_size, self.fragment_size)
        boxes = instances.bboxes
        clipped_out = np.logical_or(boxes[:, 0] == boxes[:, 2], boxes[:, 1] == boxes[:, 3])
        
        boxes = boxes[~clipped_out]
        cls = cls[~clipped_out]
        instances.update(boxes)
        
        instances.normalize(self.fragment_size, self.fragment_size)
        
        labels["instances"] = instances
        labels["cls"] = cls
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels