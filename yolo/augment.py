import cv2
import numpy as np
from ultralytics.yolo.data.augment import BaseTransform


class CropFragment(BaseTransform):
    """
    Class to crop out a fragment of an image
    """

    def __init__(
        self,
        fragment_size: int = 1280,
        gravitate_to_labels: bool = True,
        resize_shape: int = 640,
    ) -> None:
        """

        Args:
            fragment_size (int, optional): Size of the square image fragment to cout out. Defaults to 1280.
            gravitate_to_labels (bool, optional): Whether to increase the probability of gt labels within the cropped fragment. Defaults to True.
            resize_shape (int, optional): Output size the fragment is reshaped to. Defaults to 640.
        """
        super().__init__()
        self.fragment_size = fragment_size
        self.gravitate_to_labels = gravitate_to_labels
        self.resize_shape = resize_shape

    def __call__(self, labels):
        img = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(*img.shape[:2][::-1])

        # bounding positions for center of fragment
        xmin, xmax = self.fragment_size / 2, img.shape[0] - self.fragment_size / 2
        ymin, ymax = self.fragment_size / 2, img.shape[1] - self.fragment_size / 2
        boxes = instances.bboxes

        if self.gravitate_to_labels and len(boxes):
            # choose center point based on a normal distribution around
            # the average box position
            avg_box_loc = np.mean(boxes, axis=0)
            avg_x = (avg_box_loc[1] + avg_box_loc[3]) / 2
            avg_y = (avg_box_loc[0] + avg_box_loc[2]) / 2

            cx = int(round(np.random.normal(loc=avg_x, scale=img.shape[1] / 4)))
            cy = int(round(np.random.normal(loc=avg_y, scale=img.shape[0] / 4)))
            cx = np.clip(cx, xmin, xmax)
            cy = np.clip(cy, ymin, ymax)
        else:
            # choose center postion on a uniform distribution
            cx = int(round(np.random.uniform(xmin, xmax, 1)[0]))
            cy = int(round(np.random.uniform(ymin, ymax, 1)[0]))

        img = img[
            int(cx - self.fragment_size / 2) : int(cx + self.fragment_size / 2),
            int(cy - self.fragment_size / 2) : int(cy + self.fragment_size / 2),
        ]
        if (img.shape[0], img.shape[1]) != (self.resize_shape, self.resize_shape):
            img = cv2.resize(img, (self.resize_shape, self.resize_shape))

        dx = cx - self.fragment_size / 2
        dy = cy - self.fragment_size / 2

        boxes -= np.array([dy, dx] * 2)
        instances.update(boxes)
        instances.clip(self.fragment_size, self.fragment_size)
        boxes = instances.bboxes
        clipped_out = np.logical_or(
            boxes[:, 0] == boxes[:, 2], boxes[:, 1] == boxes[:, 3]
        )

        boxes = boxes[~clipped_out]
        cls = cls[~clipped_out]
        instances.update(boxes)

        instances.normalize(self.fragment_size, self.fragment_size)

        labels["instances"] = instances
        labels["cls"] = cls
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels
