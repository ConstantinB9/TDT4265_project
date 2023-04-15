import pathlib
from torch.utils.data import DataLoader
from dataset import RDDDataset
from ultralytics import yolo

from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset
from ultralytics.yolo.utils import (DEFAULT_CFG, RANK, SETTINGS, __version__,
                                    callbacks, clean_url, emojis, yaml_save)
from ultralytics.yolo.utils.checks import print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import (init_seeds, select_device)



class CustomTrainer(yolo.v8.detect.DetectionTrainer):
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
        project = self.args.project or pathlib.Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        if hasattr(self.args, 'save_dir'):
            self.save_dir = pathlib.Path(self.args.save_dir)
        else:
            self.save_dir = pathlib.Path(
                increment_path(pathlib.Path(project) / name, exist_ok=self.args.exist_ok if RANK in (-1, 0) else True))
        self.wdir = self.save_dir / 'weights'  # weights dir
        if RANK in (-1, 0):
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            yaml_save(self.save_dir / 'args.yaml', vars(self.args))  # save run args
        self.last, self.best = self.wdir / 'last.pt', self.wdir / 'best.pt'  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = self.args.model
        self.data = {
            "path": "rd2022",
            "names": {
                0: "d00",
                1: "d10",
                2: "d20",
                3: "d40"
            },
            "nc": 4
        }

        self.trainset, self.testset = "train", "val"
        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ['Loss']
        self.csv = self.save_dir / 'results.csv'
        self.plot_idx = [0, 1, 2]

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        if RANK in (-1, 0):
            callbacks.add_integration_callbacks(self)

        
    def get_dataloader(self, dataset_path, batch_size, rank=0, mode='train'):
        return DataLoader(
            dataset=RDDDataset(
                country = "Norway",
                split=dataset_path,
                split_ratio = 0.8,
                remove_empty = False        
            ),
            batch_size=batch_size
        )