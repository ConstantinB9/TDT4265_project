import json
import pathlib
import random
from itertools import repeat
from multiprocessing.pool import ThreadPool

import numpy as np
import ultralytics.yolo.data.augment as aug
from paths import data_root
from tqdm import tqdm
from ultralytics.yolo.data.dataset import YOLODataset
from ultralytics.yolo.data.utils import HELP_URL, LOGGER, get_hash
from ultralytics.yolo.utils import (
    LOCAL_RANK,
    NUM_THREADS,
    TQDM_BAR_FORMAT,
    is_dir_writeable,
)
from utils import verify_image_label


class RDDDataset(YOLODataset):
    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=None,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0,
        single_cls=False,
        use_segments=False,
        use_keypoints=False,
        data=None,
        classes=None,
        split_ratio=0.8,
    ):
        self.root_dir = data_root / "rdd2022" / "RDD2022"
        self.data_dir = self.root_dir / "Norway" / "train"
        self.split_file = self.data_dir / "split.json"

        if not self.split_file.exists():
            self.create_split(
                data_dir=self.data_dir,
                split_file=self.split_file,
                split_ratio=split_ratio,
            )
        else:
            split_data = json.load(self.split_file.open("r"))
            if split_data["split_ratio"] != split_ratio:
                self.create_split(
                    data_dir=self.data_dir,
                    split_file=self.split_file,
                    split_ratio=split_ratio,
                )

        split_data = json.load(self.split_file.open("r"))
        list_to_int = lambda iter: [int(v) for v in iter]

        self.train_ids = [int(v) for v in split_data["train"]]
        self.val_ids = [int(v) for v in split_data["val"]]
        self.id_map = {int(k): v for k, v in split_data["id_map"].items()}
        self.path_map = {int(k): v for k, v in split_data["path_map"].items()}
        super().__init__(
            img_path,
            imgsz,
            cache,
            augment,
            hyp,
            prefix,
            rect,
            batch_size,
            stride,
            pad,
            single_cls,
            use_segments,
            use_keypoints,
            data,
            classes,
        )

    @staticmethod
    def create_split(
        data_dir: pathlib.Path, split_file: pathlib.Path, split_ratio: float
    ):
        img_dir = data_dir / "images"
        all_files = [f for f in img_dir.iterdir() if f.is_file()]
        path_map = {k: v for k, v in enumerate(all_files)}
        id_map = {k: v.stem for k, v in path_map.items()}
        all_ids = list(id_map.keys())
        random.shuffle(all_ids)
        split_idx = int(round((len(all_ids) * split_ratio)))
        train_ids, val_ids = all_ids[:split_idx], all_ids[split_idx:]
        json.dump(
            {
                "split_ratio": split_ratio,
                "train": train_ids,
                "val": val_ids,
                "id_map": id_map,
                "path_map": {k: str(v) for k, v in path_map.items()},
            },
            split_file.open("w+"),
        )

    def get_img_files(self, img_path):
        ids = self.train_ids if img_path == "train" else self.val_ids
        return [self.path_map[id_] for id_ in ids]

    def get_labels(self):
        def get_lable_file(im_file):
            im_file = pathlib.Path(im_file)
            return (
                im_file.parent.parent / "annotations" / "xmls" / (im_file.stem + ".xml")
            )

        self.label_files = [get_lable_file(im_f) for im_f in self.im_files]
        cache_path = pathlib.Path(self.label_files[0]).parent / "labels.cache"
        try:
            import gc

            gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
            cache, exists = (
                np.load(str(cache_path), allow_pickle=True).item(),
                True,
            )  # load dict
            gc.enable()
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(
                [str(p) for p in self.label_files + self.im_files]
            )  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop(
            "results"
        )  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(
                None,
                desc=self.prefix + d,
                total=n,
                initial=n,
                bar_format=TQDM_BAR_FORMAT,
            )  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        if nf == 0:  # number of labels found
            raise FileNotFoundError(
                f"{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"
            )

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = (
            (len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels
        )
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            raise ValueError(
                f"All labels empty in {cache_path}, can not start training without labels. {HELP_URL}"
            )
        return labels

    def cache_labels(self, path=pathlib.Path("./labels.cache")):
        """Cache dataset labels, check images and read shapes.
        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {"labels": []}
        nm, nf, ne, nc, msgs = (
            0,
            0,
            0,
            0,
            [],
        )  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        nkpt, ndim = self.data.get("kpt_shape", (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError(
                "'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'"
            )
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(
                func=verify_image_label,  # skip verification here
                iterable=zip(
                    self.im_files,
                    self.label_files,
                    repeat(self.prefix),
                    repeat(self.use_keypoints),
                    repeat(len(self.data["names"])),
                    repeat(nkpt),
                    repeat(ndim),
                ),
            )
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for (
                im_file,
                lb,
                shape,
                segments,
                keypoint,
                nm_f,
                nf_f,
                ne_f,
                nc_f,
                msg,
            ) in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh",
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(
                f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}"
            )
        x["hash"] = get_hash([str(p) for p in self.label_files + self.im_files])
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        if is_dir_writeable(path.parent):
            if path.exists():
                path.unlink()  # remove *.cache file if exists
            np.save(str(path), x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{self.prefix}New cache created: {path}")
        else:
            LOGGER.warning(
                f"{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved."
            )
        return x

    def build_transforms(self, hyp=None):
        return aug.Compose(
            [
                aug.LetterBox(new_shape=(hyp.imgsz, hyp.imgsz), scaleFill=True),
                aug.Format(
                    bbox_format="xywh",
                    normalize=True,
                    return_mask=self.use_segments,
                    return_keypoint=self.use_keypoints,
                    batch_idx=True,
                    mask_ratio=hyp.mask_ratio,
                    mask_overlap=hyp.overlap_mask,
                )
            ]
        )
