import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, distributed
from tqdm import tqdm
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, RANK, SETTINGS,
                                    TQDM_BAR_FORMAT, __version__, callbacks,
                                    colorstr, emojis)
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import (de_parallel, select_device,
                                                smart_inference_mode)
from ultralytics.yolo.v8.detect.val import DetectionValidator

from dataset import RDDDataset


def seed_worker(worker_id):  # noqa
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CustomValidator(DetectionValidator):
    def get_dataloader(self, dataset_path, batch_size):
        mode = "val"
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        dataset = RDDDataset(
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

        workers = 8
        shuffle = mode == "train"
        batch_size = min(batch_size, len(dataset))
        nd = torch.cuda.device_count()  # number of CUDA devices
        workers = workers if mode == "train" else workers * 2
        nw = min(
            [os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers]
        )  # number of workers

        sampler = None

        # sampler = (
        #     WeightedRandomSampler(
        #         weights=dataset.get_weights(), num_samples=len(dataset)
        #     )
        #     if mode == "train"
        #     else None
        # )

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

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            self.args.half = self.device.type != "cpu"  # force FP16 val during training
            model = model.half() if self.args.half else model.float()
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots = trainer.stopper.possible_stop or (
                trainer.epoch == trainer.epochs - 1
            )
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_val_start")
            assert model is not None, "Either trainer or model is needed for validation"
            self.device = select_device(self.args.device, self.args.batch)
            self.args.half &= self.device.type != "cpu"
            model = AutoBackend(
                model,
                device=self.device,
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.model = model
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            else:
                self.device = model.device
                if not pt and not jit:
                    self.args.batch = 1  # export.py models default to batch-size 1
                    LOGGER.info(
                        f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models"
                    )

            if isinstance(self.args.data, str) and self.args.data.endswith(".yaml"):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data)
            else:
                raise FileNotFoundError(
                    emojis(
                        f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"
                    )
                )

            if self.device.type == "cpu":
                self.args.workers = (
                    0  # faster CPU val as time dominated by inference, not dataloading
                )
            if not pt:
                self.args.rect = False
            self.dataloader = self.dataloader or self.get_dataloader(
                self.data.get(self.args.split), self.args.batch
            )

            model.eval()
            model.warmup(
                imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz)
            )  # warmup

        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i

            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"])

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += trainer.criterion(preds, batch)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(
            zip(
                self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt),
            )
        )
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {
                **stats,
                **trainer.label_loss_items(
                    self.loss.cpu() / len(self.dataloader), prefix="val"
                ),
            }
            return {
                k: round(float(v), 5) for k, v in results.items()
            }  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(
            pf
            % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
        )
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(
                f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels"
            )

        # Print results per class
        for i, c in enumerate(self.metrics.ap_class_index):
            LOGGER.info(
                pf
                % (
                    self.names[c],
                    self.seen,
                    self.nt_per_class[c],
                    *self.metrics.class_result(i),
                )
            )

        if self.args.plots:
            self.confusion_matrix.plot(
                save_dir=self.save_dir, names=list(self.names.values())
            )
