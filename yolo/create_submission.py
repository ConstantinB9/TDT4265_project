import json
import pathlib
from ultralytics import YOLO
from dataset import RDDDataset

from predict import get_prediction


def main():
    """ "
    Script to create a JSON file containing all predictions on the test set in COCO format
    """
    train_idc = [19, 20]
    model_files = [
        (
            pathlib.Path(__file__).parent.parent
            / "runs"
            / "detect"
            / f'train{train_idx if train_idx else ""}'
            / "weights"
            / "last.pt"
        )
        for train_idx in train_idc
    ]

    models = [YOLO(model_file, task="detect") for model_file in model_files]

    for fragment_size in [960]:
        coco_results = []
        box_ids = 0

        for img_id, img in RDDDataset.get_test_images():
            boxes = get_prediction(
                models,
                img,
                use_fragmentation=True,
                fragment_size=fragment_size,
                intersection_th=0.3,
            )

            # h0, w0 = img.shape[:2]
            # annotator = Annotator(img)
            # for box in merged_boxes:
            #     b = box.xyxy[
            #             0
            #         ]  # get box coordinates in (top, left, bottom, right) format
            #     c = box.cls
            #     annotator.box_label(b, str(int(c.item())))
            # frame = annotator.result()
            # cv2.imwrite(f"test_predictions/pred{img_id}.jpg", frame)
            coco_data = [
                {
                    "image_id": img_id,
                    "bbox": [
                        box.xyxy[0][0].item(),
                        box.xyxy[0][1].item(),
                        box.xyxy[0][2].item() - box.xyxy[0][0].item(),
                        box.xyxy[0][3].item() - box.xyxy[0][1].item(),
                    ],
                    "category_id": int(box.cls.item()),
                    "id": box_ids + j,
                    "score": float(box.conf.item()),
                }
                for j, box in enumerate(boxes)
            ]
            box_ids += len(boxes)
            coco_results.extend(coco_data)

        json.dump(
            coco_results,
            (pathlib.Path(__file__).parent / f"submission_{fragment_size}.json").open(
                "w"
            ),
            indent=4,
        )


if __name__ == "__main__":
    main()
