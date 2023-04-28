import json
import math
import pathlib
import random
from itertools import repeat, chain
from multiprocessing.pool import ThreadPool
from typing import List

from augment import CropFragment
import cv2
import numpy as np
import ultralytics.yolo.data.augment as aug
from paths import data_root
from PIL import Image
from tqdm import tqdm
from ultralytics.yolo.data.dataset import YOLODataset
from ultralytics.yolo.data.utils import HELP_URL, LOGGER, get_hash
from ultralytics.yolo.utils import (LOCAL_RANK, NUM_THREADS, TQDM_BAR_FORMAT,
                                    is_dir_writeable)
from utils import verify_image_label, CLASS_DICT


class RDDDataset(YOLODataset):
    
    root_dir = data_root / "rdd2022" / "RDD2022"

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
        mode="train",
        pretrain=False,
        countries = None
    ):
        self.pretrain = pretrain
        if countries is None:
            countries = (
                [
                    "China_Drone",
                    "China_MotorBike",
                    "Czech",
                    "India",
                    "Japan",
                    "United_States",
                ]
                if pretrain
                else ["Norway"]
            )
        self.root_dir = data_root / "rdd2022" / "RDD2022"
        self.data_dir = [self.root_dir / country / "train" for country in countries]
        self.split_file = (
            self.data_dir[0] / "pretrain-split.json"
            if pretrain
            else self.data_dir[0] / "split_train.json"
        )
        print(self.split_file)
        self.img_cache_file = self.root_dir / f'image_cache_{"pretrain" if pretrain else ""}.npz'

        if not self.split_file.exists():
            self.create_split(
                data_dir=self.data_dir,
                split_file=self.split_file,
                split_ratio=split_ratio,
                pretrain=self.pretrain
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

        self.train_ids = [int(v) for v in split_data["train"]]
        self.val_ids = [int(v) for v in split_data["val"]]
        self.id_map = {int(k): v for k, v in split_data["id_map"].items()}
        self.path_map = {int(k): v for k, v in split_data["path_map"].items()}
        self.resize_transform = aug.LetterBox(
            new_shape=(imgsz, imgsz), scaleup=False, scaleFill=True
        )
        self.mode = mode

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
    def get_test_images(imgsz=640):
        test_root = RDDDataset.root_dir / "Norway" / "test"
        test_files = [(int(line.split(' ')[0]), line.split(' ')[1]) for line in (RDDDataset.root_dir / "Norway" / "image_ids.txt").read_text().split('\n') if line]
        for i, img_file in tqdm(test_files, desc="Loading test images"):
            img = cv2.imread(str(test_root / "images" / img_file))#[:,:,::-1]
            
            if img.size == 0:
                tqdm.write(f"ERROR: IMG {img_file} not found!")
                continue
            # img = cv2.resize(img, (imgsz, imgsz))
            yield (i, img)

    @staticmethod
    def create_split(
        data_dir: List[pathlib.Path], split_file: pathlib.Path, split_ratio: float, pretrain: bool = False
    ):
        split_dirs = data_dir if pretrain else [d for d in data_dir if d.parent.name == "Norway"]            
            
        split_files = []
        for dir in split_dirs:
            img_dir = dir / "images"
            split_dir_files = [f for f in img_dir.iterdir() if f.is_file()]
            split_files.extend(split_dir_files)

        path_map = {k: v for k, v in enumerate(split_files)}
        id_map = {k: v.stem for k, v in path_map.items()}
        all_ids = list(id_map.keys())
        random.shuffle(all_ids)
        split_idx = int(round((len(all_ids) * split_ratio)))
        train_ids, val_ids = all_ids[:split_idx], all_ids[split_idx:]
        
        exclusive_train_files = []
        train_dirs = [] if pretrain else [d for d in data_dir if d.parent.name != "Norway"]
        for dir in train_dirs:
            img_dir = dir / "images"
            train_dir_files = [f for f in img_dir.iterdir() if f.is_file()]
            exclusive_train_files.extend(train_dir_files)
        
        supp_train_paths = {k: v for k, v in enumerate(exclusive_train_files, start=max([*list(id_map.keys()), 0]) + 1)}
        path_map.update(supp_train_paths)
        id_map.update({k: v.stem for k, v in supp_train_paths.items()})
        train_ids.extend(list(supp_train_paths.keys()))

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
        if img_path == "all":
            return [self.path_map[id_] for id_ in chain(self.train_ids, self.val_ids)]
        ids = self.train_ids if img_path == "train" else self.val_ids
        return [self.path_map[id_] for id_ in ids]

    def get_labels(self):
        def get_lable_file(im_file):
            im_file = pathlib.Path(im_file)
            return (
                im_file.parent.parent / "annotations" / "xmls" / (im_file.stem + ".xml")
            )

        self.label_files = [get_lable_file(im_f) for im_f in self.im_files]
        cache_path = pathlib.Path(self.label_files[0]).parent / f"{self.mode}.cache"
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

    def cache_images(self, cache):
        """Cache images to memory or disk."""
        gb = 0  # Gigabytes of cached images
        self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni
        fcn = self.cache_images_to_disk if cache == "disk" else self.load_image
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = tqdm(
                enumerate(results),
                total=self.ni,
                bar_format=TQDM_BAR_FORMAT,
                disable=LOCAL_RANK > 0,
            )
            for i, x in pbar:
                if cache == "disk":
                    gb += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    (
                        self.ims[i],
                        self.im_hw0[i],
                        self.im_hw[i],
                    ) = x  # im, hw_orig, hw_resized = load_image(self, i)
                    gb += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({gb / 1E9:.1f}GB {cache})"
            pbar.close()

    def build_transforms(self, hyp=None):
        if self.augment:
            pre_transform = aug.Compose(
                [
                    *([CropFragment(fragment_size=1280, resize_shape=self.imgsz)] if not self.pretrain else []),
                    aug.Mosaic(
                        self,
                        imgsz=hyp.imgsz,
                        p=hyp.mosaic,
                        border=[-hyp.imgsz // 2, -hyp.imgsz // 2],
                    ),
                    aug.CopyPaste(p=hyp.copy_paste),
                    aug.RandomPerspective(
                        degrees=hyp.degrees,
                        translate=hyp.translate,
                        scale=hyp.scale,
                        shear=hyp.shear,
                        perspective=hyp.perspective,
                    ),
                ]
            )
            flip_idx = self.data.get("flip_idx", None)  # for keypoints augmentation
            if self.use_keypoints and flip_idx is None and hyp.fliplr > 0.0:
                hyp.fliplr = 0.0
                LOGGER.warning(
                    "WARNING ⚠️ No `flip_idx` provided while training keypoints, setting augmentation 'fliplr=0.0'"
                )
            transform = aug.Compose(
                [
                    pre_transform,
                    aug.MixUp(self, pre_transform=pre_transform, p=hyp.mixup),
                    aug.Albumentations(p=1.0),
                    aug.RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                    aug.RandomFlip(direction="vertical", p=hyp.flipud),
                    aug.RandomFlip(
                        direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx
                    ),
                ]
            )
        else:
            transform = aug.Compose([
                *([CropFragment(fragment_size=1280, gravitate_to_labels=True, resize_shape=self.imgsz)] if not self.pretrain else []),
                aug.LetterBox(scaleFill=True)])
        transform.append(
            aug.Format(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                return_keypoint=self.use_keypoints,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
            )
        )
        return transform

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)
                # im = cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2) if not self.pretrain else (self.imgsz, self.imgsz))
                if im is None:
                    raise FileNotFoundError(f"Image Not Found {f}")
            h0, w0 = im.shape[:2]  # orig hw
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1 and False:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(
                    im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp
                )
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def get_weights(self):
        label_counts = np.array(
            [
                [
                    int(lbl["cls"].shape[0] == 0),
                    *[int(i in lbl["cls"]) for i in range(len(CLASS_DICT))],
                ]
                for lbl in self.labels
            ]
        )
        rel_label_count = label_counts / np.sum(label_counts, axis=0)
        weights = np.max(rel_label_count, axis=1)
        return weights / np.sum(weights)


if __name__ == "__main__":
    countries = ["Norway", "China_Drone", "China_MotorBike", "Czech", "India", "Japan", "United_States"]
    coco_file = pathlib.Path().parent / "config.yaml"
    print(str(coco_file))
    from ultralytics.yolo.utils import DEFAULT_CFG
    from ultralytics.yolo.cfg import get_cfg

    hyp = get_cfg(DEFAULT_CFG, None)
    data = {
                "path": "rd2022",
                "names": {0: "d00", 1: "d10", 2: "d20", 3: "d40"},
                "nc": 4,
            }

    datasets = {c: RDDDataset(
        img_path="all",
        hyp = hyp,
        countries=[c],
        data = data,
        pretrain=True
        )
        for c in countries
        }