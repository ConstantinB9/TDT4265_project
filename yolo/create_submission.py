from itertools import islice
import json
import pathlib
import cv2
import torch
from ultralytics import YOLO
from dataset import RDDDataset
from tqdm import tqdm
from ultralytics.yolo.utils.plotting import Annotator


def main():
    train_idx = 34
    model_file = (
        pathlib.Path(__file__).parent.parent
        / "runs"
        / "detect"
        / f'train{train_idx if train_idx else ""}'
        / "weights"
        / "best.pt"
    )
    model = YOLO(model_file, task="detect")

    imgs = RDDDataset.get_test_images()
    batch_size = 64
    coco_results = []
    box_ids = 0
    batch = []
    while not batch or len(batch) == batch_size:
        batch = list(islice(imgs, batch_size))
        batch_imgs = [x[1] for x in batch]
        resized_imgs = [cv2.resize(im, (640, 640)) for im in batch_imgs]
        img_num = [x[0] for x in batch]
        results = model(resized_imgs, stream=True)
        for i, result in tqdm(
            zip(img_num, results), desc="Predicting", total=batch_size
        ):
            h0, w0 = batch_imgs[i % 64].shape[:2]
            # annotator = Annotator(batch_imgs[i % batch_size])
            # boxes = result.boxes
            # for box in boxes:
            #     b = box.xyxyn[
            #         0
            #     ]  # get box coordinates in (top, left, bottom, right) format
            #     c = box.cls
            #     annotator.box_label(torch.stack((b[0] * w0, b[1] * h0, b[2] * w0, b[3] * h0)), str(int(c.item())))
            # frame = annotator.result()
            # cv2.imwrite(f"test_predictions/pred{i}.jpg", frame)
            coco_data = [
                {
                    "image_id": i,
                    "bbox": [
                        (box[0].item() - box[2].item() / 2) * w0,
                        (box[1].item() - box[3].item() / 2) * h0,
                        box[2].item() * w0,
                        box[3].item() * h0,
                    ],
                    "category_id": int(cls.item()),
                    "id": box_ids + j,
                    "score": float(conf.item()),
                }
                for j, (box, conf, cls) in enumerate(
                    zip(result.boxes.xywhn, result.boxes.conf, result.boxes.cls)
                )
            ]
            box_ids += len(result.boxes.data)
            coco_results.extend(coco_data)

    json.dump(
        coco_results,
        (pathlib.Path(__file__).parent / "submission.json").open("w"),
        indent=4,
    )


if __name__ == "__main__":
    main()
