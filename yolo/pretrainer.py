import random

import numpy as np
import torch
from dataset import RDDDataset
from ultralytics.yolo.utils import (DEFAULT_CFG, __version__, colorstr)
from ultralytics.yolo.utils.torch_utils import de_parallel
from trainer import CustomTrainer


def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomPreTrainer(CustomTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the CustomTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        CustomTrainer.__init__(self, cfg, overrides, _callbacks)
        self.data = {
            "path": "rdd2022",
            "names": {0: "d00", 1: "d10", 2: "d20", 3: "d40"},
            "nc": 4,
        }

    def create_dataset(self, dataset_path, batch_size, rank=0, mode="train"):
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return RDDDataset(
            img_path=dataset_path,
            imgsz=self.args.imgsz,
            batch_size=batch_size,
            augment=mode == "train",  # augmentation
            hyp=self.args,  # TODO: probably add a get_hyps_from_cfg function
            rect=False,  # rectangular batches
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "),
            use_segments=self.args.task == "segment",
            use_keypoints=self.args.task == "pose",
            classes=self.args.classes,
            data=self.data,
            mode=mode,
            pretrain=True
        )
