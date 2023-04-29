import copy
import os
import pathlib
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, distributed
from ultralytics import yolo
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import (DEFAULT_CFG, RANK, SETTINGS, __version__,
                                    callbacks, colorstr, yaml_save)
from ultralytics.yolo.utils.checks import print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import (de_parallel, init_seeds,
                                                select_device)

from dataset import RDDDataset
from val import CustomValidator


def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomTrainer(yolo.v8.detect.DetectionTrainer):
    """Custom Trainer to use correct Dataset implementation.

    Args:
        yolo (_type_): _description_
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the CustomTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """
        self.args = get_cfg(cfg, overrides)
        self.device = select_device(self.args.device, self.args.batch)
        self.check_resume()
        self.validator = None
        self.model = None
        self.metrics = None
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        project = (
            self.args.project or pathlib.Path(SETTINGS["runs_dir"]) / self.args.task
        )
        name = self.args.name or f"{self.args.mode}"
        if hasattr(self.args, "save_dir"):
            self.save_dir = pathlib.Path(self.args.save_dir)
        else:
            self.save_dir = pathlib.Path(
                increment_path(
                    pathlib.Path(project) / name,
                    exist_ok=self.args.exist_ok if RANK in (-1, 0) else True,
                )
            )
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = (
            self.wdir / "last.pt",
            self.wdir / "best.pt",
        )  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == "cpu":
            self.args.workers = (
                0  # faster CPU training as time dominated by inference, not dataloading
            )

        # Model and Dataset
        self.model = self.args.model
        self.data = {
            "path": "rd2022",
            "names": {0: "d00", 1: "d10", 2: "d20", 3: "d40"},
            "nc": 4,
        }

        self.trainset, self.testset = "train", "val"
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

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
        )

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
        assert mode in ["train", "val"]
        dataset = self.create_dataset(dataset_path, batch_size, rank, mode)

        workers = 8
        shuffle = mode == "train"
        batch_size = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()  # number of CUDA devices
        workers = workers if mode == "train" else workers * 2
        nw = min(
            [os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers]
        )  # number of workers

        # sampler = (
        #     None
        #     if rank == -1
        #     else distributed.DistributedSampler(dataset, shuffle=shuffle)
        # )

        sampler = (
            WeightedRandomSampler(
                weights=dataset.get_weights(), num_samples=len(dataset)
            )
            if mode == "train"
            else None
        )

        loader = DataLoader
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)

        loader = DataLoader

        return loader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle and sampler is None,
            num_workers=nw,
            sampler=sampler,
            pin_memory=True,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            persistent_workers=(nw > 0)
            and (
                loader == DataLoader
            ),  # persist workers if using default PyTorch DataLoader
            generator=generator,
        )

    def validate(self):
        """
        Runs validation on test set using self.validator. The returned dict is expected to contain "fitness" key.
        """
        metrics = self.validator(self)
        fitness = metrics.pop(
            "fitness", -self.loss.detach().cpu().numpy()
        )  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CustomValidator(
            self.test_loader, save_dir=self.save_dir, args=copy.copy(self.args)
        )
