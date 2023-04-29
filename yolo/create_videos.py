import pathlib

import cv2
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

from predict import get_prediction

class_map = {0: "d00", 1: "d10", 2: "d20", 3: "d40"}


def main():
    """
    Script to:
    - Load video(s) frame by frame
    - Create prediction for each frame
    - Add annotations to each frame
    - Save frames as videos.
    """
    train_idc = [13, 15]
    model_files = [
        (
            pathlib.Path(__file__).parent.parent
            / "runs"
            / "detect"
            / f'train{train_idx if train_idx else ""}'
            / "weights"
            / "best.pt"
        )
        for train_idx in train_idc
    ]

    models = [YOLO(model_file, task="detect") for model_file in model_files]

    video_dir = pathlib.Path(__file__).parent.parent / "data" / "videos"
    video_file = [f for f in video_dir.iterdir() if f.is_file()]
    for file in video_file:
        vidcap = cv2.VideoCapture(str(file))
        length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(desc="Processing Frames", total=length)
        success, image = vidcap.read()
        out_imgs = []
        count = 0
        while success:
            pred = get_prediction(
                models, img=image, use_fragmentation=True, fragment_size=640
            )

            annotator = Annotator(image)
            for box in pred:
                b = box.xyxy[
                    0
                ]  # get box coordinates in (top, left, bottom, right) format
                c = box.cls
                annotator.box_label(
                    b, f"{class_map[int(c.item())]}: {box.conf.item():0.2}"
                )
            frame = annotator.result()
            out_imgs.append(frame)
            # cv2.imwrite(f"test_predictions/frame{count}.jpg", frame)
            pbar.update()
            count += 1
            success, image = vidcap.read()
            # if count > 3:
            #     break
        print("Writing video")
        out_dir = video_dir / "output"
        out_dir.mkdir(parents=False, exist_ok=True)

        frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
        fps = int(vidcap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            str(out_dir / f"{file.stem}_out.avi"),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

        for im in out_imgs:
            out.write(im)
        out.release()
        print("Writing finished")
        pbar.close()


if __name__ == "__main__":
    main()
