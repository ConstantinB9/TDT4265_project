import json
import pathlib
from ultralytics import YOLO
from dataset import RDDDataset
from tqdm import tqdm

def main():
    train_idx = 5
    model_file = pathlib.Path(__file__).parent.parent / 'runs' / 'detect' / f'train{train_idx if train_idx else ""}' / 'weights' / 'best.pt'
    model = YOLO(model_file, task="detect")
    
    
    imgs = RDDDataset.get_test_images()
    
    results = model(imgs, stream=True)
    coco_results = []
    box_ids = 0
    for i, result in tqdm(enumerate(results), desc="Predicting"):
        coco_data =[ {
            "image_id": i,
            "bbox": [
                v.item()
                for v in box[:4]
            ],
            "category_id": int(box[-1].item()),
            "id": box_ids + j,
            "score": float(box[-2].item())
            
        } for j, box in enumerate(result.boxes.data) ]
        box_ids += len(result.boxes.data)

        coco_results.extend(coco_data)
    
    json.dump(coco_results, (pathlib.Path(__file__).parent / "submission.json").open("w"), indent=4)
    

if __name__ == "__main__":
    main()